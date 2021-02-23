## Functions to complete the param structure
function initialize_param(θ, bounds, opts, _N, numerics, methods_old)
    methods = get_corrected_methods(methods_old)
    
    ind, N_diff, N_alg, N_tot = state_indices(_N, numerics)
    N = discretizations_per_section(_N.p, _N.s, _N.n, _N.a, _N.z, _N.r_p, _N.r_n, N_diff, N_alg, N_tot)
    cache = build_cache(θ, ind, N, numerics, opts, methods)

    check_errors_initial(θ, numerics, N)
    
    # while θ must contain Float64s, we need it to possibly
    # contain any type during the function generation process
    θ_any = Dict{Symbol,Any}()
    @inbounds for key in keys(θ)
        θ_any[key] = θ[key]
    end
    
    ## temporary params with no functions
    _p = param_no_funcs(θ_any, numerics, N, ind, opts, bounds, cache)
    
    if     numerics.jacobian === :symbolic
        funcs = retrieve_functions_symbolic(_p, methods)
    elseif numerics.jacobian === :AD
        funcs = load_functions(_p, methods)
    end

    # update θ
    @inbounds for method in methods
        funcs[method].update_θ!(cache.θ_tot[method], θ)
    end
    
    ## Real params with functions and methods
    p = param(
        θ,
        numerics,
        N,
        ind,
        opts,
        bounds,
        cache,
        funcs,
        methods,
    )
    return p
end

function build_cache(θ, ind, N, numerics, opts, methods)
    """
    Creates the cache struct for _Params
    """
    outputs_tot = Symbol[]
    @inbounds for (name,_type) in zip(fieldnames(typeof(ind)), fieldtypes(typeof(ind)))
        if (_type === index_state) push!(outputs_tot, name) end
    end
    outputs_tot = (outputs_tot...,)
    
    function variables_in_indices()
        """
        Defines all the possible outputs from run_model
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
    
    θ_tot = ImmutableDict{Symbol,Vector{Float64}}()
    θ_keys = ImmutableDict{Symbol,Vector{Symbol}}()

    @inbounds for method in methods
        θ_tot = ImmutableDict(θ_tot,  method => Float64[])
        θ_keys = ImmutableDict(θ_keys, method => Symbol[])
    end

    vars = variables_in_indices()
    
    opts.var_keep = model_states_logic(opts.outputs, outputs_tot)
    
    id = [
        ones(Int64, N.diff)
        zeros(Int64,N.alg)
        ]
    
    # not currently used because constraints aren't working with Sundials
    constraints = zeros(Int64, N.tot)
    constraints[ind.Φ_s] .= 1 # enforce positivity on solid phase potential in all nodes
    
    warm_start_dict = Dict{warm_start_info,Vector{Float64}}()

    cache = cache_run(
        θ_tot,
        θ_keys,
        strings_directory_func(N, numerics),
        variable_labels(),
        vars,
        outputs_tot,
        warm_start_dict,
        zeros(Float64, N.tot),
        zeros(Float64, N.tot),
        id,
        constraints,
        )
    
    return cache
end


@with_kw mutable struct state_sections{T} <: AbstractArray{T,1}
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

function retrieve_states(Y::AbstractArray, p::AbstractParam)
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
        if _type <: AbstractUnitRange && field ≠ :start && field ≠ :stop
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

    ind_alg = 0:0
    ind_diff = 0:0
    ind_tot = 0:0
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

    c_e_tot = 1:(N.p+N.s+N.n)
    c_s_avg_tot = numerics.solid_diffusion === :Fickian ? (1:N.p*N.r_p + N.n*N.r_n) : (1:(N.p+N.n))
    T_tot = numerics.temperature ? (1:(N.p+N.s+N.n) + (N.a+N.z)) : nothing
    film_tot = numerics.aging === :SEI ? (1:N.n) : nothing
    Q_tot = numerics.solid_diffusion === :polynomial ? (1:(N.p+N.n)) : nothing
    j_tot = 1:(N.p+N.n)
    j_s_tot = numerics.aging ∈ (:SEI, :R_film) ? (1:N.n) : nothing
    Φ_e_tot = 1:(N.p+N.s+N.n)
    Φ_s_tot = 1:(N.p+N.n)
    I_tot = 1
    
    
    c_e = add(:c_e,     c_e_tot,     (:p, :s, :n),         :differential)
    c_s_avg = add(:c_s_avg, c_s_avg_tot, (:p, :n),             :differential; radial = numerics.solid_diffusion === :Fickian)
    T = add(:T,       T_tot,       (:a, :p, :s, :n, :z), :differential)
    film = add(:film,    film_tot,    (:n,),                :differential)
    Q = add(:Q,       Q_tot,       (:p, :n),             :differential)
    j = add(:j,       j_tot,       (:p, :n),             :algebraic)
    j_s = add(:j_s,     j_s_tot,     (:n,),                :algebraic)
    Φ_e = add(:Φ_e,     Φ_e_tot,     (:p, :s, :n),         :algebraic)
    Φ_s = add(:Φ_s,     Φ_s_tot,     (:p, :n),             :algebraic)
    I = add(:I,       I_tot,       (),                   :algebraic)
    
    N_tot = N_diff + N_alg
    
    # These are the rest of the fields in the model_states struct that, while must be input, are unused
    Y = YP = t = V = P = SOC = index_state()
    runs = nothing
    
    ind = indices_states(Y, YP, c_e, c_s_avg, T, film, Q, j, j_s, Φ_e, Φ_s, I, t, V, P, SOC, runs)

    return ind, N_diff, N_alg, N_tot
end

@inline function guess_init(p::AbstractParam, X_applied=0.0)
    """
    Get the initial guess in the DAE initialization.
    This function is made symbolic by ModelingToolkit and saved as 
    `initial_guess.jl`
    """

    Y0 = zeros(eltype(p.θ[:c_e₀]), p.N.tot)
    YP0 = zeros(eltype(p.θ[:c_e₀]), p.N.tot)

    states = retrieve_states(Y0, p)

    build_T!(states, p)
    build_c_s_star!(states, p)
    c_s_p₀ = (p.opts.SOC*(p.θ[:θ_max_p] - p.θ[:θ_min_p]) + p.θ[:θ_min_p]) * p.θ[:c_max_p]
    c_s_n₀ = (p.opts.SOC*(p.θ[:θ_max_n] - p.θ[:θ_min_n]) + p.θ[:θ_min_n]) * p.θ[:c_max_n]
    
    function guess_differential!(states, p::AbstractParam)
        if p.numerics.solid_diffusion ∈ (:quadratic, :polynomial)
            states[:c_s_avg] .= [repeat([c_s_p₀], p.N.p); repeat([c_s_n₀], p.N.n)]
        elseif p.numerics.solid_diffusion === :Fickian
            states[:c_s_avg] .= [repeat([c_s_p₀], p.N.p*p.N.r_p); repeat([c_s_n₀], p.N.n*p.N.r_n)]
        end
        states[:c_e] = repeat([p.θ[:c_e₀]], (p.N.p+p.N.s+p.N.n))
        states[:T] = repeat([p.θ[:T₀]], (p.N.p+p.N.s+p.N.n)+p.N.a+p.N.z)
        states[:film] = zeros(p.N.n)
        states[:Q] = zeros(p.N.p+p.N.n)
    
        return nothing
    end
    
    function guess_algebraic!(states, p::AbstractParam, X_applied)
        states[:c_s_star] .= [
            repeat([c_s_p₀], p.N.p)
            repeat([c_s_n₀], p.N.n)
            ]
    
        build_OCV!(states, p)
    
        states[:j] = 0
        states[:Φ_e] = 0
        states[:Φ_s] = states[:U]
        states[:I] = X_applied
    
        if !isempty(states[:j_s]) states[:j_s] .= 0.0 end # totally arbitrary/random value for side-reaction flux
    
        return nothing
    end

    # creating initial guess vectors Y and YP
    guess_differential!(states, p)
    guess_algebraic!(states, p, X_applied)

    build_residuals!(Y0, states, p)

    return Y0, YP0
end

function strings_directory_func(N::discretizations_per_section, numerics::options_numerical; create_dir=false)

    dir_saved_models = "saved_models"

    dir_cell = "$dir_saved_models/$(Symbol(numerics.cathode))_$(Symbol(numerics.anode))"

    str = join(
        [
            "$(numerics.rxn_p)",
            "$(numerics.rxn_n)",
            "$(numerics.OCV_p)",
            "$(numerics.OCV_n)",
            "$(numerics.D_s_eff)",
            "$(numerics.rxn_rate)",
            "$(numerics.D_eff)",
            "$(numerics.K_eff)",
            "$(numerics.temperature)",
            "$(numerics.solid_diffusion)",
            "$(numerics.Fickian_method)",
            "$(numerics.aging)",
            "$(numerics.edge_values)",
            "Np$(N.p)",
            "Ns$(N.s)",
            "Nn$(N.n)",
            numerics.temperature                  ? "Na$(N.a)_Nz$(N.z)" : "",
            numerics.solid_diffusion === :Fickian ? "Nr_p$(N.r_p)_Nr_n$(N.r_n)" : ""
        ],
        "_"
    )

    dir = "$dir_cell/$str"

    if create_dir && !isdir(dir_saved_models) mkdir(dir_saved_models) end
    if create_dir && !isdir(dir_cell)         mkdir(dir_cell) end
    if create_dir && !isdir(dir)              mkdir(dir) end

    return dir
end

function strings_directory_func(p::AbstractParam; create_dir=false)
    strings_directory_func(p.N, p.numerics; create_dir=create_dir)
end

function strings_directory_func(p::AbstractParam, x; create_dir=false)
    strings_directory = string("$(strings_directory_func(p; create_dir=create_dir))/$x.jl")

    return strings_directory
end