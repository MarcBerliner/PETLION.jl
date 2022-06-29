vars_options = fieldnames(options_simulation)
vars_bounds = fieldnames(boundary_stop_conditions)[findall(fieldtypes(boundary_stop_conditions) .<: Number)]

arg_options = join(["$field = p.opts.$field" for field in vars_options], ",\n")
arg_bounds  = join(["$field = p.opts.$field" for field in vars_bounds], ",\n")

new_struct_options = join(vars_options, ",")
new_struct_bounds = join(vars_bounds, ",")

eval(quote
@inline function simulate(
    p::T1, # initial parameters file
    tf::T2 = 1e6; # a single number (length of the experiment) or a vector (interpolated points)
    sol::solution = solution(), # using a new sol or continuing from previous simulation?
    initial_states = nothing, # Starting vector of initial states
    res_I_guess = nothing,
    SOC             = p.opts.SOC, # initial SOC of the simulation. only valid if not continuing simulation
    outputs         = p.opts.outputs, # sol output states
    abstol          = p.opts.abstol, # absolute tolerance in DAE solve
    reltol          = p.opts.reltol, # relative tolerance in DAE solve
    abstol_init     = abstol, # absolute tolerance in initialization
    reltol_init     = reltol, # relative tolerance in initialization
    maxiters        = p.opts.maxiters, # maximum solver iterations
    check_bounds    = p.opts.check_bounds, # check if the boundaries (V_min, SOC_max, etc.) are satisfied
    reinit          = p.opts.reinit, # reinitialize the initial guess
    verbose         = p.opts.verbose, # print information about the run
    interp_final    = p.opts.interp_final, # interpolate the final points if a boundary is hit
    tstops          = p.opts.tstops, # times the solver explicitly solves for
    tdiscon         = p.opts.tdiscon, # times of known discontinuities in the current function
    interp_bc       = p.opts.interp_bc, # :interpolate or :extrapolate
    save_start      = p.opts.save_start, # warm-start for the initial guess
    stop_function   = p.opts.stop_function,
    calc_integrator = p.opts.calc_integrator,
    V_max         = p.bounds.V_max,
    V_min         = p.bounds.V_min,
    SOC_max       = p.bounds.SOC_max,
    SOC_min       = p.bounds.SOC_min,
    T_max         = p.bounds.T_max,
    c_s_n_max     = p.bounds.c_s_n_max,
    I_max         = p.bounds.I_max,
    I_min         = p.bounds.I_min,
    η_plating_min = p.bounds.η_plating_min,
    c_e_min       = p.bounds.c_e_min,
    dfilm_max     = p.bounds.dfilm_max,
    inputs...,
    ) where {
        T1<:model,
        T2<:Union{Number,AbstractVector,Nothing},
    }
    
    # Check if the outputs are the same as in the cache
    var_keep, outputs = solution_states_logic(outputs)

    initial_states!(sol,p,initial_states)
    
    # identifying the run type
    run = get_run(inputs,p,sol,tf)
    
    # putting opts and bounds into a structure. 
    opts = options_simulation_immutable($(Meta.parse(new_struct_options))...)
    bounds = boundary_stop_conditions_immutable($(Meta.parse(new_struct_bounds))..., boundary_stop_prev_values())
    
    # getting the initial conditions and run setup
    int, funcs, sol = initialize_simulation!(sol, p, run, bounds, opts)

    if !within_bounds(run)
        if verbose @warn "Instantly hit simulation stop conditions: $(run.info.exit_reason)" end
        exit_simulation!(p, sol, run, bounds, int, opts; cancel_interp=true)

        return sol
    end

    if verbose println("\n$(run)") end
    
    solve!(sol, int, run, p, bounds, opts, funcs)
    
    exit_simulation!(p, sol, run, bounds, int, opts)

    # If tf is a array of numbers, interpolate the results to those times
    sol = interp_sol(sol, tf, interp_bc=opts.interp_bc)

    if verbose println("\n$(sol)\n") end

    return sol
end
end)
@inline function simulate!(_sol,p::model, x...;
    outputs=isempty(_sol) ? p.opts.outputs : (@views @inbounds _sol.results[end].opts.outputs),
    overwrite_sol::Bool=false,
    sol::Nothing=nothing,
    kw...)

    simulate(p, x...;
        sol = overwrite_sol ? deepcopy(_sol) : _sol,
        outputs = outputs,
        kw...)
end

"""
If an optional input argument for the initial algebraic states is given
"""
@inline initial_states!(::solution,::model,::Nothing) = nothing
@inline function initial_states!(sol::solution,p::model,states::Vector{<:Number})
    @assert length(states) == p.N.tot
    if !isempty(sol)
        error("ERROR\n--------\n" *
        " Cannot set `initial_states` and continue a previous run.")
    end
    set_var!(sol.Y, copy(states), true)
end

@inline initial_time(sol::solution) = isempty(sol) ? 0.0 : (@inbounds nextfloat(sol.t[end]))

@inline function assess_input(inputs::T, p, sol) where T<:NamedTuple
    names = fieldnames(T)
    
    name = check_input_arguments(names)
    
    _input = redefine_func(@inbounds inputs[1])

    try
        method, input = input_method(Val(name), _input, p, sol)

        return method, input, name
    catch
        # If `input_method` is not defined for this input, return an error.
        # try/catch block let's the user dynamically add methods
        valid_methods = (method_symbol.(subtypes(AbstractMethod))...,)
        str_methods = replace("$(valid_methods)", ":"=>"")
        error("ERROR\n--------\n" *
        " Invalid keyword argument: $(String(names[1]))\n\n  Choose one from: $str_methods")
    end
end

@inline function get_run(inputs, p::model, sol::solution, tf)
    method, input, name = assess_input(values(inputs), p, sol)

    value = Ref(0.0)
    t0 = initial_time(sol)
    tf = @inbounds Float64(tf[end])
    
    run_type = run_determination(method, input)
    run = run_type(input,value,method,t0,tf,name,run_info())

    return run
end

@inline interp_sol(sol::solution,tf::T;kw...) where T<:Number = sol
@inline interp_sol(sol::solution,tf::AbstractArray{<:Number};kw...) = sol(tf;kw...)

@inline run_determination(::AbstractMethod, ::T) where T<:Any      = run_constant
@inline run_determination(::AbstractMethod, ::T) where T<:Function = run_function

@inline custom_res!(p::model,res::T,sol;kw...) where T<:Tuple = custom_res!(p,res...,sol;kw...)
@inline custom_res!(p::model,res::T,sol;kw...) where T<:Function = custom_res!(p,0.0,res,sol;kw...)
@inline function custom_res!(p::model,x::T,func_RHS::Q,sol::solution;kw...) where {T<:Function,Q<:Function}
    p.θ[:_residual_val] = 0.0
    return (t,Y,YP,p) -> func_RHS(t,Y,YP,p) - func(t,Y,YP,p)
end
@inline function custom_res!(p::model,x::T,func_RHS::Q,sol::solution;kw...) where {T<:Number,Q<:Function}
    p.θ[:_residual_val] = Float64(x)
    return func_RHS
end
@inline function custom_res!(p::model,x::T,func_RHS::Q,sol::solution;hold_val::Number=0.0,kw...) where {T<:Symbol,Q<:Function}
    if check_is_hold(x,sol)
        p.θ[:_residual_val] = hold_val
    end
    return func_RHS
end

@inline within_bounds(run::AbstractRun) = run.info.flag == -1

@inline function initialize_simulation!(sol::solution, p::model, run::T, bounds::boundary_stop_conditions_immutable, opts::AbstractOptionsModel, res_I_guess=nothing) where {method<:AbstractMethod,T<:AbstractRun{method,<:Any}}
    if !haskey(p.funcs,run)
        get_method_funcs!(p,run)
        funcs = p.funcs(run)
    end
    funcs = p.funcs(run)
    θ_tot = funcs.J_full.θ_tot
    update_θ!(θ_tot, funcs.J_full.θ_keys, p)
    
    cache = p.cache
    
    initial_guess! = p.funcs.initial_guess!
    keep_Y = opts.var_keep.Y
    
    # update the θ_tot vector from the dict p.θ
    check_errors_parameters_runtime(p, opts)
    
    # Is this a new run, or is it starting from a given initial state
    new_run = isempty(sol)
    starting_from_initial_state = new_run && !isempty(sol.Y)

    ## initializing the states vector Y and time t
    YP0 = cache.YP0
    if starting_from_initial_state
        Y0 = @inbounds sol.Y[end]
        SOC = calc_SOC(Y0, p) # estimated SOC
    elseif new_run
        Y0 = keep_Y ? zeros(Float64,length(cache.Y0)) : cache.Y0
        SOC = opts.SOC
        if opts.reinit
            initial_guess!(Y0, SOC, θ_tot, res_I_guess)
        end
    else # continue from previous simulation
        Y0 = @inbounds keep_Y ? copy(sol.Y[end]) : sol.Y[end]
        SOC = @inbounds sol.SOC[end]
    end
    
    initial_current!(Y0,YP0,p,run,sol,res_I_guess)

    ## getting the DAE integrator function
    initialize_states!(p,Y0,YP0,run,opts,funcs,SOC)

    # for new runs, check that the initial SOC is
    # within the appropriate bounds for (dis)charge
    if new_run
        I = @inbounds Y0[p.ind.I[1]]
        check_initial_SOC(bounds, SOC, I)
    end
    
    int = retrieve_integrator(run,p,funcs,Y0,YP0,opts,new_run)
    
    set_vars!(sol, p, Y0, YP0, int.t, run, opts, bounds; init_all=new_run, SOC=SOC)
    if !starting_from_initial_state
        set_var!(sol.Y, Y0, new_run || keep_Y)
    end
    
    check_simulation_stop!(sol, 0.0, Y0, YP0, run, p, bounds, opts)
    return int, funcs, sol
end

@inline function retrieve_integrator(run::T, p::model, funcs::Jac_and_res{<:Sundials.IDAIntegrator}, Y0, YP0, opts::AbstractOptionsModel, new_run::Bool) where {method<:AbstractMethod,T<:AbstractRun{method,<:Any}}
    """
    If the sol has previously been evaluated for a constant run simulation, you can reuse
    the integrator function with its cache instead of creating a new one
    """
    
    if isempty(funcs.int) || opts.calc_integrator
        int = create_integrator(run, p, funcs, Y0, YP0, opts)
    else
        # reuse the integrator cache
        int = @inbounds funcs.int[end]
        @inbounds int.p.value[] = run.value[]

        mem = int.mem
        # reinitialize at t = 0 with new Y0/YP0 and tolerances
        Sundials.IDASStolerances(mem, opts.reltol, opts.abstol)
        Sundials.IDAReInit(mem, 0.0, Y0, YP0)
        int.t = 0.0
    end

    postfix_integrator!(int, run, opts, new_run)

    return int
end

@inline function estimate_steady_state(p::model, sol::solution, opts::AbstractOptionsModel = p.opts, run::AbstractRun = (@views @inbounds sol.results[end].run);
    itermax::Int64=100)
    funcs = p.funcs(sol)
    t = @inbounds sol.t[end]
    γ = Float64(1.0)
    Y  = @inbounds copy(sol.Y[end])
    @inbounds Y[p.ind.I[1]] = 0
    YP = p.cache.YP0 .= 0
    
    R_full = funcs.R_full
    J_full = funcs.J_full
    factor = J_full.factor

    res = p.cache.Y0
    update = similar(res) .= +Inf
    J = J_full.sp

    @inbounds for iter in 1:itermax
        R_full(res,t,Y,YP,p,run)
        J_full(t,Y,YP,γ,p,run)
        factorize!(factor, J)

        res[p.ind.I[1]] = 0
        update .= factor\res
        
        Y .-= update
        if norm(update) < opts.reltol
            return Y
        elseif iter == itermax
            error("Could not converge to a steady state in $itermax iterations.")
        end
    end
end

@inline function estimate_SOC(p::model, x...;kw...)
    Y = estimate_steady_state(p, x...; kw...)
    SOC = calc_SOC(Y,p)

    return SOC
end

@inline function create_integrator(run::T, p::model, funcs::Q, Y0, YP0, opts::AbstractOptionsModel) where {T<:AbstractRun,Q<:Jac_and_res}
    R_full = funcs.R_full
    J_full = funcs.J_full
    DAEfunc = DAEFunction(
        (res,YP,Y,run,t) -> R_full(res,t,Y,YP,p,run);
        jac = (J,YP,Y,run,γ,t) -> J_full(J,t,Y,YP,γ,p,run),
        jac_prototype = J_full.sp,
    )

    prob = DAEProblem(DAEfunc, YP0, Y0, (0.0, run.tf), run, differential_vars=p.cache.id)

    int = init(prob, Sundials.IDA(;linear_solver=:KLU, init_all=false), tstops=Float64[], abstol=opts.abstol, reltol=opts.reltol, save_everystep=false, save_start=false, verbose=false)

    if isempty(funcs.int)
        push!(funcs.int, int)
    end

    return int
end
@inline function postfix_integrator!(int::Sundials.IDAIntegrator, run::AbstractRun, opts::AbstractOptionsModel, new_run::Bool)
    tstops = int.opts.tstops.valtree
    empty!(tstops)
    
    if !isempty(opts.tstops)
        append!(tstops,opts.tstops)
    end
    if !isempty(opts.tdiscon)
        append!(tstops,opts.tdiscon .- opts.reltol/2)
    end
    if !new_run
        prepend!(tstops, 1.0)
    end
    push!(tstops,run.tf)
    
    sort!(tstops)

    # the solver can fail is tstops includes 0
    if (@inbounds tstops[1]) ≤ 0.0
        deleteat!(tstops, 1:findfirst(tstops .≤ 0))
    end
    return nothing
end

@inline function solve!(sol,int::R1,run::R2,p,bounds,opts::R3,funcs) where {R1<:Sundials.IDAIntegrator,R2<:AbstractRun,R3<:AbstractOptionsModel}
    keep_Y = opts.var_keep.Y
    Y  = int.u.v
    YP = int.du.v
    
    iter = 1
    status = within_bounds(run)
    @inbounds while status
        step!(int)
        iter += 1
        t = int.t

        set_vars!(sol, p, Y, YP, t, run, opts, bounds)
        
        check_simulation_stop!(sol, t, Y, YP, run, p, bounds, opts)
        
        status = check_solve(run, sol, int, p, bounds, opts, funcs, keep_Y, iter, Y, t)
    end
    
    run.info.iterations = iter
    return nothing
end

@inline function exit_simulation!(p::R1, sol::R2, run::R3, bounds::R4, int::R5, opts::R6; cancel_interp::Bool=false) where {R1<:model,R2<:solution,R3<:AbstractRun,R4<:boundary_stop_conditions_immutable,R5<:Sundials.IDAIntegrator,R6<:AbstractOptionsModel}
    # if a stop condition (besides t = tf) was reached
    if !cancel_interp
        if opts.interp_final && !(run.info.flag == 0) && int.t > 1
            interp_final_points!(p, sol, run, bounds, int, opts)
        else
            set_var!(sol.Y, opts.var_keep.Y ? copy(int.u.v) : int.u.v, opts.var_keep.Y)
        end
    end

    iterations_start = isempty(sol) ? 0 : (@inbounds @views sol.results[end].run_index[end])

    run_index = (1:run.info.iterations) .+ iterations_start

    tspan = (run.t0, @inbounds sol.t[end])

    results = run_results(
        run,
        tspan,
        run.info,
        run_index,
        int,
        opts,
        bounds,
        p.N,
        p.numerics,
        p,
    )

    push!(sol.results, results)

    return nothing
end

@views @inbounds @inline function interp_final_points!(p::R1, sol::R2, run::R3, bounds::R4, int::R5, opts::R6) where {R1<:model,R2<:solution,R3<:AbstractRun,R4<:boundary_stop_conditions_immutable,R5<:Sundials.IDAIntegrator,R6<:AbstractOptionsModel}
    if opts.var_keep.YP
        YP = length(sol.YP) > 1 ? bounds.prev.t_final_interp_frac.*(sol.YP[end] .- sol.YP[end-1]) .+ sol.YP[end-1] : sol.YP[end]
    else
        YP = Float64[]
    end
    
    t = bounds.prev.t_final_interp_frac*(int.t - int.tprev) + int.tprev
    
    set_var!(sol.Y,  bounds.prev.t_final_interp_frac.*(int.u.v .- sol.Y[end]) .+ sol.Y[end], opts.var_keep.Y)
    set_vars!(sol, p, sol.Y[end], YP, t, run, opts, bounds; modify! = set_var_last!)
    
    return nothing
end

@inline function save_start_init!(Y0::Vector{Float64}, run::AbstractRun, p::model, SOC::Float64)
    key = save_start_info(
        run.method,
        round(SOC, digits=4),
        round(value(run), digits=4),
    )

    save_start_dict = p.cache.save_start_dict
    
    key_exists = key ∈ keys(save_start_dict)
    if key_exists
        @inbounds Y0[(1:p.N.alg) .+ p.N.diff] .= save_start_dict[key]
    end
    
    return key, key_exists
end
@inline initialize_states!(p::model, Y0::T, YP0::T, run::AbstractRun, opts::AbstractOptionsModel, funcs::Jac_and_res; kw...) where {T<:Vector{Float64}} = newtons_method!(p,Y0,YP0,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg;kw...)
@inline function initialize_states!(p::model, Y0::T, YP0::T, run::AbstractRun, opts::AbstractOptionsModel, funcs::Jac_and_res, SOC::Number;kw...) where {T<:Vector{Float64}}
    if opts.save_start
        key, key_exists = save_start_init!(Y0, run, p, SOC)
        
        newtons_method!(p,Y0,YP0,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg;kw...)
        
        if !key_exists
            p.cache.save_start_dict[key] = @inbounds Y0[(1:p.N.alg) .+ p.N.diff]
        end
    else
        newtons_method!(p,Y0,YP0,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg;kw...)
    end
    
    return nothing
end

@inline factorize!(factor::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64},A::SparseMatrixCSC{Float64, Int64}) = LinearAlgebra.lu!(factor,A)
@inline function factorize!(factor::KLUFactorization{Float64, Int64},A::SparseMatrixCSC{Float64, Int64})
    klu!(factor,A)
    
    #=
    # If the condition number is too large, then a refactor may be necessary
    cond_est = 1.0/rcond(factor)
    if cond_est > 1e12
        klu_factor!(factor)
    end
    =#
end

@inline function newtons_method!(p::model,Y,YP,run::AbstractRun,opts=p.opts;kw...)
    funcs = p.funcs(run)
    newtons_method!(p,Y,YP,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg;kw...)
end
@inline function newtons_method!(p::model,Y::R1,YP::R1,run,opts::AbstractOptionsModel,R_alg::T1,R_diff::T2,J_alg::T3;
    itermax::Int64=100, t::Float64=0.0
    ) where {R1<:Vector{Float64},T1<:residual_combined,T2<:residual_combined,T3<:jacobian_combined}

    res    = p.cache.res
    Y_old  = p.cache.Y_alg
    Y_new  = @views @inbounds Y[p.N.diff+1:end]
    YP    .= 0.0
    J      = J_alg.sp
    γ      = 0.0
    factor = J_alg.factor
    
    # starting loop for Newton's method
    @inbounds for iter in 1:itermax
        # updating res, Y, and J
        R_alg(res,t,Y,YP,p,run)
        J_alg(t,Y,YP,γ,p,run)
        factorize!(factor, J)
        
        Y_old .= Y_new
        Y_new .-= factor\res
        if norm(Y_old .- Y_new) < opts.reltol_init # || maximum(abs, res) < opts.abstol_init
            break
        elseif iter == itermax
            error("Could not initialize DAE in $itermax iterations.")
        end
    end
    # calculate the differential equations for YP0
    R_diff(YP,t,Y,YP,p,run)
    
    # Estimate dY_alg/dt. Improves stability of the initial guess
    Δt = max(10opts.reltol_init, sqrt(eps(p.θ[:c_e₀])))
    
    Y_new_time = p.cache.Y_full
    Y_new_time .= Y .+ Δt*YP
    
    R_alg(res,Δt,Y_new_time,YP,p,run)
    
    # Update the algebraic Jacobian (currently not used)
    # J_alg(Δt,Y_new_time,YP,γ,p,run)
    # factorize!(factor, J)

    # The Newton update `-(factor\res)` is equal to (Y_alg(Δt + t0) - Y_alg(t0))
    @inbounds YP[p.N.diff+1:end] .= -(factor\res)./Δt
    
    return nothing
end
