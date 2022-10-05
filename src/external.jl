## Functions to complete the model structure
function petlion(cathode::Function;
    load_funcs::Bool = true,
    kwargs...,
    )
    θ = OrderedDict{Symbol,Float64}()
    funcs = _funcs_numerical()

    anode, system = cathode(θ,funcs)
    if anode isa Function && system isa Function
        anode(θ,funcs)
        θ, bounds, opts, N, numerics = system(θ, funcs, cathode, anode; kwargs...)

        return initialize_param(θ, bounds, opts, N, numerics, load_funcs)
    else
        error("Cathode function must return both anode and system functions.")
    end
end
function petlion(;
    cathode=cathode,
    anode=anode,
    system=system, # Input chemistry - can be modified
    load_funcs::Bool=true,
    kwargs... # keyword arguments for system
    )
    θ = OrderedDict{Symbol,Float64}()
    funcs = _funcs_numerical()

    cathode(θ, funcs)
    anode(θ, funcs)
    θ, bounds, opts, N, numerics = system(θ, funcs; kwargs...)

    p = initialize_param(θ, bounds, opts, N, numerics, load_funcs)

    return p
end

function initialize_param(θ, bounds, opts, _N, numerics, load_funcs=true)
    
    ind, N_diff, N_alg, N_tot = state_indices(_N, numerics)
    N = discretizations_per_section(_N.p, _N.s, _N.n, _N.a, _N.z, _N.r_p, _N.r_n, N_diff, N_alg, N_tot)
    cache = build_cache(θ, ind, N, numerics, opts)

    check_errors_initial(θ, numerics, N)
    
    # while θ must contain Float64s, we need it to possibly
    # contain any type during the function generation process
    θ_any = convert(OrderedDict{Symbol,Any}, θ)
    θ[:I1C] = calc_I1C(θ)
    
    ## temporary params with no functions
    _p = model_skeleton(θ_any, numerics, N, ind, opts, bounds, cache)
    
    funcs = load_funcs ? load_functions(_p) : model_funcs()

    sort!(θ)
    
    ## Real params with functions
    p = model(
        θ,
        numerics,
        N,
        ind,
        opts,
        bounds,
        cache,
        funcs,
    )
    return p
end

function build_cache(θ, ind, N, numerics, opts)
    """
    Creates the cache struct for _petlion
    """
    outputs_tot = Symbol[]
    @inbounds for (name,_type) in zip(fieldnames(typeof(ind)), fieldtypes(typeof(ind)))
        if !(_type <: Tuple) push!(outputs_tot, name) end
    end
    outputs_tot = (outputs_tot...,)
    
    function variables_in_indices()
        """
        Defines all the possible outputs from simulate
        """
        
        var = Symbol[]
        tot = Int64[]
        
        for field in outputs_tot
            ind_var = getproperty(ind, field)
            if ind_var.var_type ∈ (:differential, :algebraic)
                push!(var, field)
                push!(tot, ind_var.start)
            end
        end
        (var[sortperm(tot)]...,)
    end

    function variable_labels()
        """
        Returns a vector of labels for all the states. Same length as Y and YP
        """
        labels = Symbol[]
        
        function add_label!(var)
            x = getproperty(ind, var)
            sections = x.sections
            if !isempty(x.sections)
                for section in sections
                    for i in 1:length(getproperty(x, section))
                        push!(labels, Symbol(var, :_, section, i))
                    end
                end
            else
                push!(labels, var)
            end
        end
        
        for var in variables_in_indices()
            x = getproperty(ind, var)
            x.var_type ∈ (:differential,:algebraic) ? add_label!(var) : nothing
        end

        @assert length(labels) == N.tot
        
        return labels
    end

    θ_tot =  Float64[]
    θ_keys =  Symbol[]

    vars = variables_in_indices()
    
    Y0 = zeros(Float64, N.tot)
    Y_full = zeros(Float64, N.tot)
    YP0 = zeros(Float64, N.tot)
    res = zeros(Float64, N.alg)
    Y_alg = zeros(Float64, N.alg)
    
    opts.var_keep = solution_states_logic(opts.outputs)[1]
    outputs_possible = Symbol[
        :Y
        :YP
        :t
        keys(merge(model_states_and_outputs(numerics; remove_inactive=true)...))...
    ]

    id = [
        ones(Int64, N.diff)
        zeros(Int64,N.alg)
        ]
    
    save_start_dict = Dict{save_start_info,Vector{Float64}}()

    cache = cache_run(
        θ_tot,
        θ_keys,
        variable_labels(),
        vars,
        save_start_dict,
        Y0,
        Y_full,
        YP0,
        res,
        Y_alg,
        outputs_possible,
        id,
        )
    
    return cache
end

Base.@kwdef mutable struct state_sections{T} <: AbstractArray{T,1}
    tot::AbstractArray{T,1} = nothing
    a = nothing
    p = nothing
    s = nothing
    n = nothing
    z = nothing
    sections::Tuple = ()
    var_type::Symbol = :NA
end
Base.size(state::state_sections) = size(state.tot)
Base.IndexStyle(::Type{<:state_sections}) = IndexLinear()
Base.getindex(state::state_sections, i::Int64)= state.tot[i]
Base.setindex!(state::state_sections, v, i::Int64)= (state.tot[i] = v)
is_active(s::state_sections) = !isempty(s)

function retrieve_states(Y::AbstractArray, p::AbstractModel)
    """
    Creates a dictionary of variables based on an input array and the indices in p.ind
    """
    if length(Y) < p.N.tot error("Input vector must be ≥ the number of indices.") end

    states = Dict{Symbol,Any}()
    ind = p.ind
    vars_in_use = p.cache.vars
    
    vars = fieldnames(solution)[findall(fieldtypes(solution) .<: AbstractArray{<:Number})]

    sections = Symbol[]
    @inbounds for (field,_type) in zip(fieldnames(index_state), fieldtypes(index_state))
        if _type <: AbstractUnitRange && !(field ∈ (:start,:stop))
            push!(sections, field)
        end
    end
    sections = (sections...,)
    
    @views @inbounds function add_state!(var)
        ind_var = getproperty(ind, var)
        
        section_values = Any[]
        
        for section in sections
            ind_section = getproperty(ind_var, section)

            push!(section_values, section ∈ ind_var.sections ? Y[ind_section] : nothing)
        end
        
        state = state_sections(
            var ∈ vars_in_use ? Y[ind_var] : [],
            section_values...,
            ind_var.sections,
            ind_var.var_type,
        )
        
        states[var] = state

        return nothing
    end

    @inbounds for var in vars
        add_state!(var)
    end
    
    return states
end

function state_new(x, sections::Tuple, p;
        var_type=:NA,
        section_length = [getproperty(p.N, section) for section in sections],
    )
    
    x_vec = Any[]
    start = 0
    @inbounds for (i,section) in enumerate(sections)
        N = section_length[i]
        ind = (1:N) .+ start
        push!(x_vec, @views @inbounds x[ind])
        
        start += N
    end
    
    state = state_sections(;
        tot = x,
        NamedTuple{sections}((x_vec...,))...,
        sections = sections,
        var_type = var_type,
    )
    
    return state
end

function state_indices(N, numerics)
    """
    Creates the struct which has the indices of all the states.
    There can be more items in the struct than the simulation actually uses – 
    this is because states are needed despite not being a function in the DAE,
    e.g., temperature is still used in isothermal simulations
    """

    # preallocate the state dict with empty index_state structs
    state_dict = OrderedDict{Symbol,Any}(
        [name for (name,field) in zip(fieldnames(indices_states),fieldtypes(indices_states)) if field == index_state_immutable] .=> Ref(index_state())
    )
    
    function add!(var::Symbol,var_type::Symbol,::Tuple{})
        state_dict[var] = index_state(;start=1, stop=1, var_type=var_type)
    end
    function add!(var::Symbol,var_type::Symbol,sections::NTuple{T,Symbol}) where T
        """
        Adds a state to the indices dictionary
        """
        @assert !(:particle_p ∈ sections && :p ∈ sections)
        @assert !(:particle_n ∈ sections && :n ∈ sections)
        
        all_sections = (:a,:p,:s,:n,:z)
        active_sections = Symbol[]

        len = 0
        indices = Dict(all_sections .=> Ref(0:0))
        for section in sections
            if section == :particle_p
                ind = 1:(N.p*N.r_p)
                push!(active_sections, :p)
            elseif section == :particle_n
                ind = 1:(N.n*N.r_n)
                push!(active_sections, :n)
            else
                ind = 1:getfield(N, section)
                push!(active_sections, section)
            end
            ind = ind .+ len
            indices[active_sections[end]] = ind

            len += length(ind)
        end

        active_sections_ordered = Tuple(intersect(all_sections,active_sections))

        state_dict[var] = index_state(1,
            len,
            getindex.(Ref(indices),all_sections)...,
            active_sections_ordered,
            var_type,
        )
    end
    
    #### Define the states and the sections they are in ####
    active_states = model_states_and_outputs(numerics; remove_inactive=true)[1]
    for state in keys(active_states)
        active_state = active_states[state]
        add!(state, active_state.var_type, active_state.sections)
    end

    N_type = Dict((:differential,:algebraic,:tot) .=> 0)

    state_vars = Symbol[]
    for var_type in (:differential, :algebraic), state in keys(state_dict)
        state_ind = state_dict[state]
        if state_dict[state].var_type == var_type
            push!(state_vars, state)
            tot = N_type[:tot]

            for x in [:start, :stop, state_ind.sections...]
                setfield!(state_ind, x, getfield(state_ind, x) .+ tot)
            end

            N_type[var_type] += length(state_dict[state])
            N_type[:tot] += length(state_dict[state])
        end
    end

    ind = indices_states(
        [index_state_immutable(state_dict[state]) for state in keys(state_dict)]...,
        Tuple(state_vars),
    )

    N_diff = N_type[:differential]
    N_alg  = N_type[:algebraic]
    N_tot  = N_type[:tot]

    return ind, N_diff, N_alg, N_tot
end

function indices_section(sections::Tuple, p::AbstractModel; offset::Int64=0)
    """
    Retrieve the indices of the given sections.
    """
    
    ind_start = offset
    sections_full = fieldnames(index_state)[findall(fieldtypes(index_state) .<: UnitRange)]
    
    order_section = sort([findfirst(sections_full .== section) for section in sections])
    sections_organized = (sections_full[order_section]...,)

    indices = Vector{UnitRange{Int64}}(undef, length(sections_organized))
    for (i,section) in enumerate(sections_organized)
        N = getfield(p.N, section)
        indices[i] = (1:N) .+ ind_start
        ind_start += N
    end

    ind = index_state(;
        start = 1,
        stop = ind_start,
        NamedTuple{sections_organized}(indices)...,
        sections = sections_organized,
    )
    
    return ind
end

model_info(p::AbstractModel) = model_info(p.N, p.numerics)
function model_info(N::T1,numerics::T2) where {T1<:discretizations_per_section,T2<:options_numerical}
    version = "PETLION version: v"*join(Symbol.(PETLION_VERSION),".")

    numerical = ["$field: $(getproperty(numerics,field))" for field in fieldnames(T2)]
    
    discretization = [
        "N.p: $(N.p)";
        "N.s: $(N.s)";
        "N.n: $(N.n)";
        numerics.temperature                  ? "N.a: $(N.a)\nN.z: $(N.z)" : "";
        numerics.solid_diffusion == :Fickian ? "N.r_p: $(N.r_p)\nN.r_n: $(N.r_n)" : "";
        ]
    filter!(!isempty,discretization)
    
    str  = version * "\n"^2
    str *= "options_numerical\n" * join(numerical, "\n") * "\n"^2
    str *= "discretizations_per_section\n" * join(discretization, "\n")

    return str
end

function strings_directory_func(N::discretizations_per_section, numerics::T; create_dir=false) where T<:options_numerical

    dir_saved_models = options[:DIR_SAVED_MODELS]
    # If the file directory is not specified, use the current working directory
    if !isnothing(options[:FILE_DIRECTORY])
        dir_saved_models = joinpath(options[:FILE_DIRECTORY], dir_saved_models)
    end
    
    if create_dir && !isdir(dir_saved_models)
        mkdir(dir_saved_models)
    end

    dir_cell = "$dir_saved_models/$(Symbol(numerics.cathode))_$(Symbol(numerics.anode))"

    str = join(
        [
            ["$(getproperty(numerics,field))" for field in filter(x->!(x ∈ (:cathode,:anode)), fieldnames(T))];
            "Np$(N.p)";
            "Ns$(N.s)";
            "Nn$(N.n)";
            numerics.temperature                  ? "Na$(N.a)_Nz$(N.z)" : "";
            numerics.solid_diffusion == :Fickian ? "Nr_p$(N.r_p)_Nr_n$(N.r_n)" : "";
        ],
        "_"
    )

    str = Base.bytes2hex(sha1(str))

    dir = "$dir_cell/$str"

    if create_dir
        for x in (dir_saved_models, dir_cell, dir)
            if !isdir(x)
                mkdir(x)
            end
        end
    end
    
    return dir
end

function strings_directory_func(p::AbstractModel; create_dir=false)
    strings_directory_func(p.N, p.numerics; create_dir=create_dir)
end

function strings_directory_func(p::AbstractModel, x; kw...)
    strings_directory = string("$(strings_directory_func(p; kw...))/$x.jl")

    return strings_directory
end


@inline function trapz(x::T1,y::T2) where {T1<:AbstractVector{<:Number},T2<:AbstractVector{<:Number}}
    """
    Trapezoidal rule with SIMD vectorization
    """
    @assert length(x) == length(y)
    out = 0.0
    @inbounds @simd for i in 2:length(x)
        out += 0.5*(x[i] - x[i-1])*(y[i] + y[i-1])
    end
    return out
end
@inline function cumtrapz(x::T1,y::T2) where {T1<:AbstractVector{<:Number},T2<:AbstractVector{<:Number}}
    # Check matching vector length
    @assert length(x) == length(y)
    
    # Initialize Output
    out = similar(x)
    out[1] = 0

    # Iterate over arrays
    @inbounds for i in 2:length(x)
        out[i] = out[i-1] + 0.5*(x[i] - x[i-1])*(y[i] + y[i-1])
    end
    # Return output
    return out
end

function extrap_x_0(x::AbstractVector,y::AbstractVector)
    @inbounds y[1] - ((y[3] - y[1] - (((x[2] - x[1])^-1)*(x[3] - x[1])*(y[2] - y[1])))*((x[3]^2 - (x[1]^2) - (((x[2] - x[1])^-1)*(x[2]^2 - (x[1]^2))*(x[3] - x[1])))^-1)*(x[1]^2)) - ((y[2] - y[1] - (((x[3]^2 - (x[1]^2) - (((x[2] - x[1])^-1)*(x[2]^2 - (x[1]^2))*(x[3] - x[1])))^-1)*(x[2]^2 - (x[1]^2))*(y[3] - y[1] - (((x[2] - x[1])^-1)*(x[3] - x[1])*(y[2] - y[1])))))*((x[2] - x[1])^-1)*x[1])
end
function extrapolate_section(y::AbstractVector,p::AbstractModel,section::Symbol)
    """
    Extrapolate to the edges of the FVM sections with a second-order polynomial
    """
    
    N = getproperty(p.N, section)
    
    x_range = [
        0
        collect(range(1/2N,1-1/2N,length=N))
        1
    ]
    
    x_interp = @views @inbounds x_range[2:4]
    
    y_range = [
        extrap_x_0(x_interp,y)
        y
        extrap_x_0(x_interp,(@inbounds @views y[end:-1:end-2]))
    ]
    
    x_range *= p.θ[Symbol(:l_,section)]
    
    return x_range, y_range
end