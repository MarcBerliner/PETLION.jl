function emptyfunc end

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

function load_functions_symbolic(p::AbstractParam, method::Symbol, YP_cache=nothing)
    dir = strings_directory_func(p) * "/$method/"

    ## residuals
    initial_guess!  = include(dir * "initial_guess.jl")
    f!              = include(dir * "f.jl")
    f_alg!          = include(dir * "f_alg.jl")
    f_diff!         = include(dir * "f_diff.jl")

    ## Jacobian
    BSON.@load dir * "J_sp.jl" J_y_sp θ_keys θ_len
    
    J_y!_func     = include(dir * "J_y.jl")
    J_y_alg!_func = include(dir * "J_y_alg.jl")

    J_y!_sp     = sparse(J_y_sp...)
    J_y_alg!_sp = J_y!_sp[p.N.diff+1:end,p.N.diff+1:end]
    
    J_y!      = jacobian_symbolic(J_y!_func, J_y!_sp)
    J_y_alg!  = jacobian_symbolic(J_y_alg!_func, J_y_alg!_sp)

    return initial_guess!, f!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys, θ_len
end

function load_functions(p::AbstractParam, methods::Tuple)
    
    if     p.numerics.jacobian === :symbolic
        jac_type  = jacobian_symbolic
        load_func = load_functions_symbolic
    elseif p.numerics.jacobian === :AD
        jac_type  = jacobian_AD
        load_func = load_functions_forward_diff
    end

    ## Pre-allocation
    Y0_alg      = zeros(Float64, p.N.alg)
    Y0_alg_prev = zeros(Float64, p.N.alg)
    res         = zeros(Float64, p.N.alg)
    Y0_diff     = zeros(Float64, p.N.diff)
    
    ## Begin loading functions based on method
    funcs = ImmutableDict{Symbol, functions_model{jac_type}}()
    
    @inbounds for method in methods
        initial_guess!, f!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys, θ_len = load_func(p, method, Y0_diff)

        update_θ! = update_θ_maker(θ_keys, θ_len)
        append!(p.cache.θ_keys[method], θ_keys)
        append!(p.cache.θ_tot[method],  zeros(sum(θ_len)))

        initial_conditions = init_newtons_method(f_alg!, J_y_alg!, f_diff!, Y0_alg, Y0_alg_prev, Y0_diff, res)
        funcs = ImmutableDict(funcs, method => functions_model{jac_type}(f!, initial_guess!, J_y!, initial_conditions, update_θ!))
    end
    
    return funcs
end



function get_corrected_methods(methods)
    """
    Corrects the input methods to `Params`
    """
    if methods isa Symbol
        methods = (methods,)
    elseif isempty(methods)
        error("methods cannot be empty")
    end
    check_appropriate_method.(methods)
    
    return methods
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
    
    θ_tot  = ImmutableDict{Symbol,Vector{Float64}}()
    θ_keys = ImmutableDict{Symbol,Vector{Symbol}}()

    @inbounds for method in methods
        θ_tot  = ImmutableDict(θ_tot,  method => Float64[])
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

function retrieve_states(Y::AbstractArray, p::AbstractParam)
    """
    Creates a dictionary of variables based on an input array and the indices in p.ind
    """
    if length(Y) < p.N.tot error("Input vector must be ≥ the number of indices.") end
    
    states = Dict{Symbol,Any}()
    ind    = p.ind
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
            ind_section  = getproperty(ind_var, section)

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

    ind_alg  = 0:0
    ind_diff = 0:0
    ind_tot  = 0:0
    N_diff = 0
    N_alg  = 0

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
        stop  = tot[end]

        return index_state(start, stop, a, p, s, n, z, (sections...,), var_type)
    end

    c_e_tot     = 1:(N.p+N.s+N.n)
    c_s_avg_tot = numerics.solid_diffusion === :Fickian ? (1:N.p*N.r_p + N.n*N.r_n) : (1:(N.p+N.n))
    T_tot       = numerics.temperature ? (1:(N.p+N.s+N.n) + (N.a+N.z)) : nothing
    film_tot    = numerics.aging === :SEI ? (1:N.n) : nothing
    Q_tot       = numerics.solid_diffusion === :polynomial ? (1:(N.p+N.n)) : nothing
    j_tot       = 1:(N.p+N.n)
    j_s_tot     = numerics.aging ∈ (:SEI, :R_film) ? (1:N.n) : nothing
    Φ_e_tot     = 1:(N.p+N.s+N.n)
    Φ_s_tot     = 1:(N.p+N.n)
    I_tot       = 1
    
    
    c_e      = add(:c_e,     c_e_tot,     (:p, :s, :n),         :differential)
    c_s_avg  = add(:c_s_avg, c_s_avg_tot, (:p, :n),             :differential; radial = numerics.solid_diffusion === :Fickian)
    T        = add(:T,       T_tot,       (:a, :p, :s, :n, :z), :differential)
    film     = add(:film,    film_tot,    (:n,),                :differential)
    Q        = add(:Q,       Q_tot,       (:p, :n),             :differential)
    j        = add(:j,       j_tot,       (:p, :n),             :algebraic)
    j_s      = add(:j_s,     j_s_tot,     (:n,),                :algebraic)
    Φ_e      = add(:Φ_e,     Φ_e_tot,     (:p, :s, :n),         :algebraic)
    Φ_s      = add(:Φ_s,     Φ_s_tot,     (:p, :n),             :algebraic)
    I        = add(:I,       I_tot,       (),                   :algebraic)
    
    N_tot  = N_diff + N_alg
    
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

    Y0  = zeros(eltype(p.θ[:c_e₀]), p.N.tot)
    YP0 = zeros(eltype(p.θ[:c_e₀]), p.N.tot)

    states = retrieve_states(Y0, p)

    build_T!(states, p)
    build_c_s_star!(states, p)
    
    function guess_differential!(states, p::AbstractParam)

        c_s_p₀, c_s_n₀ = surfaceConcentrationInitial(p)
        if p.numerics.solid_diffusion ∈ (:quadratic, :polynomial)
            states[:c_s_avg] .= [repeat([c_s_p₀], p.N.p); repeat([c_s_n₀], p.N.n)]
        elseif p.numerics.solid_diffusion === :Fickian
            states[:c_s_avg] .= [repeat([c_s_p₀], p.N.p*p.N.r_p); repeat([c_s_n₀], p.N.n*p.N.r_n)]
        end
        states[:c_e]  = repeat([p.θ[:c_e₀]], (p.N.p+p.N.s+p.N.n))
        states[:T]    = repeat([p.θ[:T₀]], (p.N.p+p.N.s+p.N.n)+p.N.a+p.N.z)
        states[:film] = zeros(p.N.n)
        states[:Q]    = zeros(p.N.p+p.N.n)
    
        return nothing
    
    end
    
    function guess_algebraic!(states, p::AbstractParam, X_applied)
        c_s_p₀, c_s_n₀ = surfaceConcentrationInitial(p)
    
        states[:c_s_star] .= [
            repeat([c_s_p₀], p.N.p)
            repeat([c_s_n₀], p.N.n)
            ]
    
        build_OCV!(states, p)
    
        states[:j]   = 0
        states[:Φ_e] = 0
        states[:Φ_s] = states[:U]
        states[:I]   = X_applied
    
        if !isempty(states[:j_s]) states[:j_s] .= 0.0 end # totally arbitrary/random value for side-reaction flux
    
        return nothing
    end

    # creating initial guess vectors Y and YP
    guess_differential!(states, p)
    guess_algebraic!(states, p, X_applied)

    build_residuals!(Y0, states, p)

    return Y0, YP0
end

@inline function set_vars!(model::R1, p::R2, Y::R3, YP::R3, t::R4, run::R5, opts::R6; init_all::Bool=false, modify!::Function=set_var!) where {R1<:model_output, R2<:param, R3<:Vector{Float64}, R4<:Float64, R5<:AbstractRun, R6<:options_model}

    ind  = p.ind
    keep = opts.var_keep

    # these variables must be calculated, but they may not necessarily be kept
    modify!(model.t,        (keep.t       || init_all), t + run.t0 )
    modify!(model.V,        (keep.V       || init_all), calc_V(Y, p, run) )
    modify!(model.I,        (keep.I       || init_all), calc_I(Y, model, run, p) )
    modify!(model.P,        (keep.P       || init_all), calc_P(Y, model, run, p) )
    modify!(model.c_s_avg,  (keep.c_s_avg || init_all), @views @inbounds Y[ind.c_s_avg] )
    modify!(model.SOC,      (keep.SOC     || init_all), @views @inbounds calc_SOC(model.c_s_avg[end], p) )
    
    # these variables do not need to be calculated
    if keep.YP  modify!(model.YP,  (keep.YP  || init_all), (keep.YP ? copy(YP) : YP)   ) end
    if keep.c_e modify!(model.c_e, (keep.c_e || init_all), @views @inbounds Y[ind.c_e] ) end
    if keep.j   modify!(model.j,   (keep.j   || init_all), @views @inbounds Y[ind.j]   ) end
    if keep.Φ_e modify!(model.Φ_e, (keep.Φ_e || init_all), @views @inbounds Y[ind.Φ_e] ) end
    if keep.Φ_s modify!(model.Φ_s, (keep.Φ_s || init_all), @views @inbounds Y[ind.Φ_s] ) end
    
    # exist as an optional output if the model uses them
    if ( p.numerics.temperature                    && keep.T )    modify!(model.T,    (keep.T    || init_all), @views @inbounds Y[ind.T]    ) end
    if ( p.numerics.aging === :SEI                 && keep.film ) modify!(model.film, (keep.film || init_all), @views @inbounds Y[ind.film] ) end
    if ( !(p.numerics.aging === false)             && keep.j_s )  modify!(model.j_s,  (keep.j_s  || init_all), @views @inbounds Y[ind.j_s]  ) end
    if ( p.numerics.solid_diffusion === :quadratic && keep.Q )    modify!(model.Q,    (keep.Q    || init_all), @views @inbounds Y[ind.Q]    ) end

    return nothing
end

@inline function set_var!(x::T1, append::Bool, x_val::T2) where {T1<:Vector{Float64},T2<:Float64}
    if append
        push!(x, x_val)
    else
        @views @inbounds x[1] = x_val
    end
end
@inline function set_var!(x::T1, append::Bool, x_val::T2) where {T1<:VectorOfArray{Float64,2,Array{Array{Float64,1},1}},T2<:AbstractVector}
    if append
        push!(x, x_val)
    else
        @views @inbounds x[1] .= x_val
    end
end

@inline function set_var_last!(x::T1, append::Bool, x_val::T2) where {T1<:Vector{Float64},T2<:Float64}
    @views @inbounds x[end] = x_val
end
@inline function set_var_last!(x::T1, append::Bool, x_val::T2) where {T1<:VectorOfArray{Float64,2,Array{Array{Float64,1},1}},T2<:AbstractVector}
    @views @inbounds x[end] .= x_val
end

function strings_directory_func(N::discretizations_per_section, numerics::options_numerical; create_dir=false)

    dir_saved_models = "PET_jl_saved_models"

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

function sparsejacobian_multithread(ops::AbstractVector{<:Num}, vars::AbstractVector{<:Num};
    sp = ModelingToolkit.jacobian_sparsity(ops, vars),
    simplify = true,
    show_progress = true,
    multithread = true,
    )

    I,J,_ = findnz(sp)

    exprs = Vector{Num}(undef, length(I))

    iters = show_progress ? ProgressBars.ProgressBar(1:length(I)) : 1:length(I)
    if multithread
        @inbounds Threads.@threads for iter in iters
            @inbounds exprs[iter] = expand_derivatives(Differential(vars[J[iter]])(ops[I[iter]]), simplify)
        end
    else
        @inbounds for iter in iters
            @inbounds exprs[iter] = expand_derivatives(Differential(vars[J[iter]])(ops[I[iter]]), simplify)
        end
    end

    jac = sparse(I,J, exprs, length(ops), length(vars))
    return jac
end

function _symbolic_initial_guess(p::AbstractParam, SOC_sym, θ_sym, I_current_sym)
    
    Y0_sym = guess_init(p, +1.0)[1]
    Y0_sym[p.ind.I] .= I_current_sym

    return Y0_sym
end

function _symbolic_residuals(p::AbstractParam, t_sym, x_sym, xp_sym, I_current_sym, θ_sym, method)
    ## symbolic battery model
    res = zeros(eltype(t_sym), size(p.cache.Y0))
    residuals_PET!(res, t_sym, x_sym, xp_sym, method, p)

    if method ∈ (:I, :V, :P)
        ind_res = 1:length(res)-1
    else
        ind_res = 1:length(res)
    end

    return res, ind_res
end
function _Jacobian_sparsity_pattern(p, res, x_sym, xp_sym)
    sp_x  = ModelingToolkit.jacobian_sparsity(res, x_sym)
    sp_xp = ModelingToolkit.jacobian_sparsity(res, xp_sym)
    
    I,J,_ = findnz(sp_x .+ sp_xp)

    J_sp = sparse(I, J, 1.0, p.N.tot, p.N.tot)
    
    return J_sp, sp_x, sp_xp
end

function _symbolic_jacobian(p::AbstractParam, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method;
    res_prev=nothing, Jac_prev=nothing, verbose=false)

    J_sp, sp_x, sp_xp = _Jacobian_sparsity_pattern(p, res, x_sym, xp_sym)

    # if this function has previously been evaluated, this will make the next one quicker.
    # Don't bother recomputing the Jacobian for lines of the residual that are identical
    flag_prev = !isnothing(res_prev) && !isnothing(Jac_prev)
    if flag_prev
        ind_new = Int64[]
        @inbounds Threads.@threads for i in 1:length(res)
            if @views @inbounds !isequal(res[i], res_prev[i])
                push!(ind_new, i)
                SparseArrays.fkeep!(Jac_prev, (I,J,K) -> I ≠ i)
            end
        end

    else
        ind_new = 1:length(res)
    end

    ## symbolic jacobian
    Jacxp = sparsejacobian_multithread(res, xp_sym; show_progress = false, sp = sp_xp)
    function check_semi_explicit(Jacxp)
        I,J,K = findnz(Jacxp)
        @inbounds for (i,j,k) in zip(I,J,K)
            @assert i === j # only along the diagonal
            @assert isequal(k,1) || isequal(k,-1) # only ones or negative ones on diagonal
        end
    end
    check_semi_explicit(Jacxp)

    Jac_new = @inbounds @views sparsejacobian_multithread(res[ind_new], x_sym;  show_progress = !flag_prev)
    
    # For some reason, Jac[ind_new] .= Jac_new doesn't work on linux. This if statement is a temporary workaround
    if !flag_prev
        Jac = Jac_new
    else
        Jac = Jac_prev
        Jac[ind_new,:] .= Jac_new
    end

    """
    Sometimes, the differential terms do not appear in the row corresponding to their value.
    this line ensures that there is still a spot for the ∂x/∂t term in the jacobian sparsity pattern/
    functon, otherwise there will be a mismatch
    """
    @inbounds for i in 1:p.N.diff
        if !sp_x[i,i]
            Jac[i,i] = 1e-100 # some arbitrarily small number so it doesn't get overwritten
        end
    end
    
    @assert length(Jac.nzval) === length(J_sp.nzval)

    # building the sparse jacobian
    _, jacFunc  = build_function(Jac, x_sym, θ_sym_slim, fillzeros=true, checkbounds=false)
    
    return Jac, jacFunc, J_sp
end
function _symbolic_initial_conditions_res(p::AbstractParam, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method, ind_res)
    
    res_diff = res[1:p.N.diff]
    res_alg  = zeros(eltype(res), p.N.alg)
    res_alg[ind_res[p.N.diff+1:end] .- p.N.diff] .= res[ind_res[p.N.diff+1:end]]
    
    _, res_diffFunc = build_function(res_diff, x_sym, xp_sym, θ_sym_slim, fillzeros=true, checkbounds=false)
    _, res_algFunc  = build_function(res_alg,  x_sym[p.N.diff+1:end], x_sym[1:p.N.diff], θ_sym_slim, fillzeros=true, checkbounds=false)
    
    return res_algFunc, res_diffFunc
end
function _symbolic_initial_conditions_jac(p::AbstractParam, Jac, x_sym, xp_sym, θ_sym, θ_sym_slim, method)
    
    jac_alg  = Jac[p.N.diff+1:end,p.N.diff+1:end]
    
    _, jac_algFunc  = build_function(jac_alg,  x_sym[p.N.diff+1:end], x_sym[1:p.N.diff], θ_sym_slim, fillzeros=true, checkbounds=false)
    
    return jac_algFunc
end

function get_only_θ_used_in_model(θ_sym, res, Y0_sym, θ_keys, θ_len)
    """
    Some of the parameters in θ may not be used in the model. This returns
    a vector of the ones that are actually used and their sorted names
    """
    used_params_tot = eltype(θ_sym)[]
    @inbounds for (Y,r) in zip(Y0_sym,res)
        append!(used_params_tot, ModelingToolkit.get_variables(Y))
        append!(used_params_tot, ModelingToolkit.get_variables(r))
    end

    index_params = Int64[]

    dummy = BitArray{1}(undef, length(θ_sym))
    @inbounds for x in unique(used_params_tot)
        dummy .= isequal.(x, θ_sym)
        if sum(dummy) === 1
            index_param = findfirst(dummy)
            push!(index_params, index_param)
        end
    end
    sort!(index_params)

    θ_sym_slim  = θ_sym[index_params]
    
    # remove additional entries due to parameters that are vectors
    θ_keys_extend = Symbol[]
    θ_len_extend  = Int64[]
    @inbounds for (key,len) in zip(θ_keys, θ_len)
        append!(θ_keys_extend, repeat([key], len))
        append!(θ_len_extend, repeat([len], len))
    end
    
    θ_keys_slim = θ_keys_extend[index_params]
    θ_len_slim  = θ_len_extend[index_params]
    
    # Vectors of length n will have n-1 duplicates. This removes them
    unique_ind  = indexin(unique(θ_keys_slim), θ_keys_slim)
    θ_keys_slim = θ_keys_slim[unique_ind]
    θ_len_slim  = θ_len_slim[unique_ind]
    
    return θ_sym_slim, θ_keys_slim, θ_len_slim
end
function update_θ_maker(θ_vars::AbstractVector, θ_len::Vector{Int64})
    str = "((x,θ) -> (@inbounds @views begin;"

    start = 0
    @inbounds for (θ, l) in zip(θ_vars, θ_len)
        if l === 1
            str *= "x[$(1+start)]=θ[:$(θ)]::Float64;"
        else
            str *= "x[$((1:l).+start)].=θ[:$(θ)]::Vector{Float64};"
        end
        start += l
    end
    str *= "end;nothing))"

    func = mk_function(Meta.parse(str));
    return func
end

function get_symbolic_vars(p::AbstractParam)
    θ_keys = sort!(Symbol.(keys(p.θ))) # sorted for convenience
    θ_len = Int64[length(p.θ[key]) for key in θ_keys]

    ModelingToolkit.@variables x_sym[1:p.N.tot], xp_sym[1:p.N.tot], t_sym, SOC_sym, I_current_sym
    ModelingToolkit.@parameters θ_sym[1:sum(θ_len)]

    p_sym = deepcopy(p)
    p_sym.opts.SOC = SOC_sym

    start = 0
    @inbounds for (i,key) in enumerate(θ_keys)
        if θ_len[i] === 1
            p_sym.θ[key] = θ_sym[1+start]
        else
            p_sym.θ[key] = θ_sym[(1:θ_len[i]) .+ start]
        end

        start += θ_len[i]
    end

    return θ_sym, x_sym, xp_sym, t_sym, SOC_sym, I_current_sym, p_sym, θ_keys, θ_len
end

function load_functions_forward_diff(p::AbstractParam, method::Symbol, YP_cache::Vector{Float64})

    θ_sym, x_sym, xp_sym, t_sym, SOC_sym, I_current_sym, p_sym, θ_keys, θ_len = get_symbolic_vars(p)

    ## Y0 function
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, I_current_sym)

    ## batteryModel function
    res, _ = _symbolic_residuals(p_sym, t_sym, x_sym, xp_sym, I_current_sym, θ_sym, method)

    ind_res = 1:length(res)
    
    θ_sym_slim, θ_keys_slim, θ_len_slim = get_only_θ_used_in_model(θ_sym, res, Y0_sym, θ_keys, θ_len)
    
    _, Y0Func  = build_function(Y0_sym, SOC_sym, θ_sym_slim, I_current_sym, fillzeros=true, checkbounds=false)
    _, resFunc = build_function(res, x_sym, xp_sym, θ_sym_slim, fillzeros=true, checkbounds=false)
    
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method, ind_res)

    J_y_sp, sp_x, sp_xp = _Jacobian_sparsity_pattern(p, res, x_sym, xp_sym)

    J_y_sp_alg = J_y_sp[p.N.diff+1:end,p.N.diff+1:end]

    function build_color_jacobian_struct(J, f!, N, _YP_cache=zeros(Float64,N))
        colorvec = SparseDiffTools.matrix_colors(J)

        Y_cache = zeros(Float64, N)

        func = res_FD(f!, _YP_cache, p.cache.θ_tot[method])
        
        jac_cache = SparseDiffTools.ForwardColorJacCache(
            func,
            Y_cache,
            dx       = similar(Y_cache),
            colorvec = colorvec,
            sparsity = similar(J),
            )

        J_struct = jacobian_AD(func, J, jac_cache)
        return J_struct
    end

    initial_guess! = eval(Y0Func)
    f!             = eval(resFunc)
    f_alg!         = eval(res_algFunc)
    f_diff!        = eval(res_diffFunc)

    J_y!     = build_color_jacobian_struct(J_y_sp, f!, p.N.tot)
    J_y_alg! = build_color_jacobian_struct(J_y_sp_alg, f_alg!, p.N.alg, YP_cache)


    @assert size(J_y!.sp)     === (p.N.tot,p.N.tot)
    @assert size(J_y_alg!.sp) === (p.N.alg,p.N.alg)

    return initial_guess!, f!, f_alg!, f_diff!, J_y!, J_y_alg!, θ_keys_slim, θ_len_slim
end

function generate_functions_symbolic(p::AbstractParam, method::Symbol;
    verbose=true,
    # if this function has previously been evaluated, this will make the next one quicker
    res_prev=nothing,
    Jac_prev=nothing,
    )
    
    println_v(x...) = verbose ? println(x...) : nothing

    if isnothing(res_prev) && isnothing(Jac_prev)
        println_v("Creating the functions for method $method, $p")
    else
        println_v("Continuing to create method $method")
    end

    θ_sym, x_sym, xp_sym, t_sym, SOC_sym, I_current_sym, p_sym, θ_keys, θ_len = get_symbolic_vars(p)

    ## Y0 function
    println_v("Making initial guess function...")
    Y0_sym = _symbolic_initial_guess(p_sym, SOC_sym, θ_sym, I_current_sym)
    println_v("Done\n")

    ## batteryModel function
    println_v("Making symbolic model...")
    res, ind_res = _symbolic_residuals(p_sym, t_sym, x_sym, xp_sym, I_current_sym, θ_sym, method)
    println_v("Done\n")

    θ_sym_slim, θ_keys_slim, θ_len_slim = get_only_θ_used_in_model(θ_sym, res, Y0_sym, θ_keys, θ_len)
    
    _, Y0Func  = build_function(Y0_sym, SOC_sym, θ_sym_slim, I_current_sym, fillzeros=true, checkbounds=false)
    res_build = zeros(eltype(res), p.N.tot)
    res_build[ind_res] .= res[ind_res]
    _, resFunc = build_function(res_build, x_sym, xp_sym, θ_sym_slim, fillzeros=true, checkbounds=false)

    ## jacobian
    println_v("Making symbolic Jacobian. May take a few mins...")
    Jac, jacFunc, J_sp = _symbolic_jacobian(p, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method;
        res_prev=res_prev, Jac_prev=Jac_prev, verbose=verbose)

    println_v("Done\n")

    println_v("Making initial condition functions...")
    res_algFunc, res_diffFunc = _symbolic_initial_conditions_res(p, res, x_sym, xp_sym, θ_sym, θ_sym_slim, method, ind_res)

    jac_algFunc= _symbolic_initial_conditions_jac(p, Jac, x_sym, xp_sym, θ_sym, θ_sym_slim, method)
    
    dir = strings_directory_func(p; create_dir=true) * "/$method/"
    mkdir(dir)
        
    write(dir * "initial_guess.jl", string(Y0Func))
    write(dir * "f.jl",             string(resFunc))
    write(dir * "J_y.jl",           string(jacFunc))
    write(dir * "f_alg.jl",         string(res_algFunc))
    write(dir * "J_y_alg.jl",       string(jac_algFunc))
    write(dir * "f_diff.jl",        string(res_diffFunc))
        
    J_y_sp = (findnz(J_sp)..., p.N.tot, p.N.tot)
    
    θ_keys = θ_keys_slim
    θ_len = θ_len_slim
    BSON.@save dir * "J_sp.jl" J_y_sp θ_keys θ_len

    println_v("Finished\n")

    res_prev = res
    Jac_prev = Jac

    return res_prev, Jac_prev
end

function retrieve_functions_symbolic(p::AbstractParam, methods::Tuple)
    ## Checking if the main directory exists
    dir = strings_directory_func(p)
    
    ## Checking if the methods exist
    res_prev = Jac_prev = nothing
    @inbounds for method in methods
        if !isdir(dir * "/$method/")
            res_prev, Jac_prev = generate_functions_symbolic(p, method, res_prev=res_prev, Jac_prev=Jac_prev)
        end
    end

    funcs = load_functions(p, methods)
    
    return funcs
end

function retrieve_functions_forward_diff(p::AbstractParam, methods::Tuple)
    ## Checking if the main directory exists
    dir = strings_directory_func(p)
    
    @inbounds for method in methods
        generate_functions_forward_diff(p, method)
    end
    
    return funcs
end

@inline function calc_I1C(p::param)
    F = 96485.3365
    θ = p.θ

    if p.numerics.aging === :R_film
        @inbounds @views I1C = (F/3600.0)*min(
            θ[:c_max_n]::Float64*(θ[:θ_max_n] - θ[:θ_min_n]::Float64)*(1.0 - (θ[:ϵ_n][1])::Float64 - θ[:ϵ_fn]::Float64)*θ[:l_n]::Float64,
            θ[:c_max_p]::Float64*(θ[:θ_min_p] - θ[:θ_max_p]::Float64)*(1.0 - θ[:ϵ_p]::Float64      - θ[:ϵ_fp]::Float64)*θ[:l_p]::Float64,
        )
    else
        @inbounds @views I1C = (F/3600.0)*min(
            θ[:c_max_n]::Float64*(θ[:θ_max_n] - θ[:θ_min_n]::Float64)*(1.0 - θ[:ϵ_n]::Float64 - θ[:ϵ_fn]::Float64)*θ[:l_n]::Float64,
            θ[:c_max_p]::Float64*(θ[:θ_min_p] - θ[:θ_max_p]::Float64)*(1.0 - θ[:ϵ_p]::Float64 - θ[:ϵ_fp]::Float64)*θ[:l_p]::Float64,
        )
    end

    return I1C
end

@inline function calc_V(Y::Vector{Float64}, p::param, run::AbstractRun, ind_Φ_s::T=p.ind.Φ_s) where {T<:AbstractUnitRange{Int64}}
    if run.method === :V
        V = value(run)
    else
        if p.numerics.edge_values === :center
            V = @views @inbounds Y[ind_Φ_s[1]] - Y[ind_Φ_s[end]]
        else # interpolated edges
            V = @views @inbounds (1.5*Y[ind_Φ_s[1]] - 0.5*Y[ind_Φ_s[2]]) - (1.5*Y[ind_Φ_s[end]] - 0.5*Y[ind_Φ_s[end-1]])
        end
    end
    return V
end

@inline function calc_I(Y::Vector{Float64}, model::model_output, run::AbstractRun, p::param)
    if run.method === :I
        I = @inbounds value(run)
    elseif  run.method === :V
        I = @inbounds Y[p.ind.I][1]
    elseif run.method === :P
        I = @inbounds value(run)/model.V[end]
    end
    
    return I
end

@inline function calc_P(Y::Vector{Float64}, model::model_output, run::AbstractRun, p::param)
    if run.method === :I || run.method === :V
        P = @views @inbounds model.I[end]*model.V[end]
    elseif run.method === :P
        P = value(run)
    end

    return P
end

@inline function calc_SOC(c_s_avg::Vector{Float64}, p::param)
    if p.numerics.solid_diffusion === :Fickian
        c_s_avg_sum = @views @inbounds mean(c_s_avg[(p.N.p*p.N.r_p)+1:end])
    else # c_s_avg in neg electrode
        c_s_avg_sum = @views @inbounds mean(c_s_avg[p.N.p+1:end])
    end

    return (c_s_avg_sum/p.θ[:c_max_n]::Float64 - p.θ[:θ_min_n]::Float64)/(p.θ[:θ_max_n]::Float64 - p.θ[:θ_min_n]::Float64) # cell-soc fraction
end

@inline function interpolate_model(model::R1, tspan::T1, interp_bc::Symbol) where {R1<:model_output,T1<:Union{Number,AbstractVector}}
    dummy = similar(model.t)

    if tspan isa UnitRange
        t = collect(tspan)
    elseif tspan isa Real
        t = Float64[tspan]
    else
        t = tspan
    end

    f(x) = interpolate_variable(x, model, t, dummy, interp_bc)
    
    # collect all the variables for interpolation
    states_tot = Any[]
    @inbounds for field in fieldnames(model_output)
        if field === :t
            push!(states_tot, tspan)
        else
            x = getproperty(model, field)
            if x isa AbstractArray{Float64} && length(x) > 1
                push!(states_tot, f(x))
            else
                push!(states_tot, x)
            end
        end
        
    end

    model = R1(states_tot...)

    return model
end
@inline function interpolate_variable(x::R1, model::R2, tspan::T1, dummy::Vector{Float64}, interp_bc::Symbol) where {R1<:Vector{Float64},R2<:model_output,T1<:Union{Real,AbstractArray}}
    spl = Dierckx.Spline1D(model.t, x; bc = (interp_bc == :interpolate ? "nearest" : (interp_bc == :extrapolate ? "extrapolate" : error("Invalid interp_bc method."))))
    out = spl(tspan)
    
    return out
end

@inline function interpolate_variable(x::R1, model::R2, tspan::T1, dummy::Vector{Float64}, interp_bc::Symbol) where {R1<:Union{VectorOfArray{Float64,2,Array{Array{Float64,1},1}},Vector{Vector{Float64}}},R2<:model_output,T1<:Union{Real,AbstractArray}}
    @inbounds out = [copy(x[1]) for _ in tspan]

    @inbounds for i in eachindex(x[1])

        @inbounds for j in eachindex(x)
            @inbounds dummy[j] = x[j][i]
        end

        spl = Dierckx.Spline1D(model.t, dummy; bc = (interp_bc == :interpolate ? "nearest" : (interp_bc == :extrapolate ? "extrapolate" : error("Invalid interp_bc method."))))

        @inbounds for (j,t) in enumerate(tspan)
            @inbounds out[j][i] = spl(t)
        end

    end

    return VectorOfArray(out)
end



