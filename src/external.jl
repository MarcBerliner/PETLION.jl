## Functions to complete the model structure
function petlion(cathode::Function;kwargs...)
    θ = Dict{Symbol,Float64}()
    funcs = _funcs_numerical()

    anode, system = cathode(θ,funcs)
    if anode isa Function && system isa Function
        anode(θ,funcs)
        θ, bounds, opts, N, numerics = system(θ, funcs, cathode, anode; kwargs...)

        return initialize_param(θ, bounds, opts, N, numerics)
    else
        error("Cathode function must return both anode and system functions.")
    end
end
function petlion(;cathode=cathode,anode=anode,system=system, # Input chemistry - can be modified
    kwargs... # keyword arguments for system
    )
    θ = Dict{Symbol,Float64}()
    funcs = _funcs_numerical()

    cathode(θ, funcs)
    anode(θ, funcs)
    θ, bounds, opts, N, numerics = system(θ, funcs; kwargs...)

    p = initialize_param(θ, bounds, opts, N, numerics)

    return p
end

function initialize_param(θ, bounds, opts, _N, numerics)
    
    ind, N_diff, N_alg, N_tot = state_indices(_N, numerics)
    N = discretizations_per_section(_N.p, _N.s, _N.n, _N.a, _N.z, _N.r_p, _N.r_n, N_diff, N_alg, N_tot)
    cache = build_cache(θ, ind, N, numerics, opts)

    check_errors_initial(θ, numerics, N)
    
    # while θ must contain Float64s, we need it to possibly
    # contain any type during the function generation process
    θ_any = Dict{Symbol,Any}()
    @inbounds for key in keys(θ)
        θ_any[key] = θ[key]
    end
    θ[:I1C] = calc_I1C(θ)
    
    ## temporary params with no functions
    _p = model_skeleton(θ_any, numerics, N, ind, opts, bounds, cache)
    
    funcs = load_functions(_p)
    
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
        if (_type === index_state) push!(outputs_tot, name) end
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
        
        @assert length(labels) === N.tot
        
        return labels
    end

    θ_tot =  Float64[]
    θ_keys =  Symbol[]

    vars = variables_in_indices()
    
    opts.var_keep = model_states_logic(opts.outputs, outputs_tot)

    Y0 = zeros(Float64, N.tot)
    YP0 = zeros(Float64, N.tot)
    res = zeros(Float64, N.alg)
    Y_alg = zeros(Float64, N.alg)
    
    id = [
        ones(Int64, N.diff)
        zeros(Int64,N.alg)
        ]
    
    # not currently used because constraints aren't working with Sundials
    constraints = zeros(Int64, N.tot)
    constraints[ind.Φ_s] .= 1 # enforce positivity on solid phase potential in all nodes
    
    save_start_dict = Dict{save_start_info,Vector{Float64}}()

    cache = cache_run(
        θ_tot,
        θ_keys,
        strings_directory_func(N, numerics),
        variable_labels(),
        vars,
        outputs_tot,
        save_start_dict,
        Y0,
        YP0,
        res,
        Y_alg,
        id,
        constraints,
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

function retrieve_states(Y::AbstractArray, p::AbstractModel)
    """
    Creates a dictionary of variables based on an input array and the indices in p.ind
    """
    if length(Y) < p.N.tot error("Input vector must be ≥ the number of indices.") end
    
    states = Dict{Symbol,Any}()
    ind = p.ind
    vars_in_use = p.cache.vars
    
    vars = p.cache.outputs_tot

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
            section_name = Symbol(var, :_, section)
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

    N_diff = 0
    N_alg = 0

    function add(var::Symbol, tot, vars::Tuple, var_type::Symbol=:NA;
        radial::Bool = false, replace = 0:0)
        """
        Adds a state to the indices dictionary
        """
        if tot === nothing
            return index_state()
        elseif tot isa Int
            tot = tot:tot
        end
        
        if var_type ∈ (:differential, :algebraic)
            tot = tot .+ (N_diff+N_alg)
            if     var_type === :differential
                N_diff += length(tot)
            elseif var_type === :algebraic
                N_alg  += length(tot)
            end
        end
        
        ind_a = 1:N.a
        ind_p = 1:(radial ? N.p*N.r_p : N.p)
        ind_s = 1:N.s
        ind_n = 1:(radial ? N.n*N.r_n : N.n)
        ind_z = 1:N.z

        sections = Symbol[]
        ind_start = 0
        :a ∈ vars ? (a = tot[ind_a .+ ind_start]; push!(sections, :a); ind_start += length(ind_a)) : a = replace
        :p ∈ vars ? (p = tot[ind_p .+ ind_start]; push!(sections, :p); ind_start += length(ind_p)) : p = replace
        :s ∈ vars ? (s = tot[ind_s .+ ind_start]; push!(sections, :s); ind_start += length(ind_s)) : s = replace
        :n ∈ vars ? (n = tot[ind_n .+ ind_start]; push!(sections, :n); ind_start += length(ind_n)) : n = replace
        :z ∈ vars ? (z = tot[ind_z .+ ind_start]; push!(sections, :z); ind_start += length(ind_z)) : z = replace

        start = tot[1]
        stop = tot[end]

        return index_state(start, stop, a, p, s, n, z, (sections...,), var_type)
    end

    c_e_tot     = 1:(N.p+N.s+N.n)
    c_s_avg_tot = numerics.solid_diffusion === :Fickian ? (1:N.p*N.r_p + N.n*N.r_n) : (1:(N.p+N.n))
    T_tot       = numerics.temperature ? (1:(N.p+N.s+N.n) + (N.a+N.z)) : nothing
    film_tot    = numerics.aging === :SEI ? (1:N.n) : nothing
    Q_tot       = numerics.solid_diffusion === :polynomial ? (1:(N.p+N.n)) : nothing
    j_tot       = 1:(N.p+N.n)
    j_s_tot     = numerics.aging ∈ (:SEI, :R_aging) ? (1:N.n) : nothing
    SOH_tot     = numerics.aging ∈ (:SEI, :R_aging) ? 1 : nothing
    Φ_e_tot     = 1:(N.p+N.s+N.n)
    Φ_s_tot     = 1:(N.p+N.n)
    I_tot       = 1
    
    c_e     = add(:c_e,     c_e_tot,     (:p, :s, :n),         :differential)
    c_s_avg = add(:c_s_avg, c_s_avg_tot, (:p, :n),             :differential; radial = numerics.solid_diffusion === :Fickian)
    T       = add(:T,       T_tot,       (:a, :p, :s, :n, :z), :differential)
    film    = add(:film,    film_tot,    (:n,),                :differential)
    Q       = add(:Q,       Q_tot,       (:p, :n),             :differential)
    SOH     = add(:SOH,     SOH_tot,     (),                   :differential)
    j       = add(:j,       j_tot,       (:p, :n),             :algebraic)
    j_s     = add(:j_s,     j_s_tot,     (:n,),                :algebraic)
    Φ_e     = add(:Φ_e,     Φ_e_tot,     (:p, :s, :n),         :algebraic)
    Φ_s     = add(:Φ_s,     Φ_s_tot,     (:p, :n),             :algebraic)
    I       = add(:I,       I_tot,       (),                   :algebraic)
    
    N_tot = N_diff + N_alg
    
    # These are the rest of the fields in the model_states struct that, while must be input, are unused
    Y = YP = t = V = P = SOC = index_state()
    runs = nothing
    
    ind = indices_states(Y, YP, c_e, c_s_avg, T, film, Q, j, j_s, Φ_e, Φ_s, I, t, V, P, SOC, SOH, runs)

    return ind, N_diff, N_alg, N_tot
end

@inline function guess_init(p::AbstractModel, X_applied=0.0)
    """
    Get the initial guess in the DAE initialization.
    This function is made symbolic by Symbolics and saved as 
    `initial_guess.jl`
    """

    Y0  = zeros(eltype(p.θ[:c_e₀]), p.N.tot)
    YP0 = zeros(eltype(p.θ[:c_e₀]), p.N.tot)

    states = retrieve_states(Y0, p)
    
    build_T!(states, p)

    states[:c_s_avg].p .= p.θ[:c_max_p] * (p.opts.SOC*(p.θ[:θ_max_p] - p.θ[:θ_min_p]) + p.θ[:θ_min_p])
    states[:c_s_avg].n .= p.θ[:c_max_n] * (p.opts.SOC*(p.θ[:θ_max_n] - p.θ[:θ_min_n]) + p.θ[:θ_min_n])
    
    build_c_s_star!(states, p)

    build_OCV!(states, p)
    
    # differential
    states[:c_e] = repeat([p.θ[:c_e₀]], (p.N.p+p.N.s+p.N.n))
        
    states[:T] = repeat([p.θ[:T₀]], (p.N.p+p.N.s+p.N.n)+p.N.a+p.N.z)
        
    states[:film] = zeros(p.N.n)
        
    states[:Q] = zeros(p.N.p+p.N.n)

    if p.numerics.aging ∈ (:SEI, :R_aging) states[:SOH] = 1.0 end
    
    # algebraic
    states[:j] = 0.0
        
    states[:Φ_e] = 0.0
        
    states[:Φ_s] = states[:U]
        
    states[:I] = X_applied
    
    if !isempty(states[:j_s]) states[:j_s] .= 0.0 end

    build_residuals!(Y0, states, p)

    return Y0, YP0
end

model_info(p::AbstractModel) = model_info(p.N, p.numerics)
function model_info(N::T1,numerics::T2) where {T1<:discretizations_per_section,T2<:options_numerical}
    version = "PETLION version: v"*join(Symbol.(VERSION),".")

    numerical = ["$field: $(getproperty(numerics,field))" for field in fieldnames(T2)]
    
    discretization = [
        "N.p: $(N.p)";
        "N.s: $(N.s)";
        "N.n: $(N.n)";
        numerics.temperature                  ? "N.a: $(N.a)\nN.z: $(N.z)" : "";
        numerics.solid_diffusion === :Fickian ? "N.r_p: $(N.r_p)\nN.r_n: $(N.r_n)" : "";
        ]
    filter!(!isempty,discretization)
    
    str  = version * "\n"^2
    str *= "options_numerical\n" * join(numerical, "\n") * "\n"^2
    str *= "discretizations_per_section\n" * join(discretization, "\n")

    return str
end

function strings_directory_func(N::discretizations_per_section, numerics::T; create_dir=false) where T<:options_numerical

    dir_saved_models = "saved_models"
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
            numerics.solid_diffusion === :Fickian ? "Nr_p$(N.r_p)_Nr_n$(N.r_n)" : "";
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


@inline function trapz(x::T1,y::T2) where {T1<:AbstractVector,T2<:AbstractVector}
    """
    Trapezoidal rule with SIMD vectorization
    """
    @assert length(x) === length(y)
    out = 0.0
    @inbounds @simd for i in 2:length(x)
        out += 0.5*(x[i] - x[i-1])*(y[i] + y[i-1])
    end
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