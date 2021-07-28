@inline function run_model(
    p::T1, # initial parameters file
    tspan::T2 = nothing; # a single number (length of the experiment) or a vector (interpolated points)
    model::R1 = model_output(), # using a new model or continuing from previous simulation?
    I = nothing, # constant current C-rate. also accepts :rest
    V = nothing, # constant voltage. also accepts :hold if continuing simulation
    P = nothing, # constant power.   also accepts :hold if continuing simulation
    res = nothing, # custom residual function
    SOC::Number = p.opts.SOC, # initial SOC of the simulation. only valid if not continuing simulation
    outputs::R5 = p.opts.outputs, # model output states
    abstol::Float64 = p.opts.abstol, # absolute tolerance in DAE solve
    reltol::Float64 = p.opts.reltol, # relative tolerance in DAE solve
    abstol_init::Float64 = abstol, # absolute tolerance in initialization
    reltol_init::Float64 = reltol, # relative tolerance in initialization
    maxiters::Int64 = p.opts.maxiters, # maximum solver iterations
    check_bounds::Bool = p.opts.check_bounds, # check if the boundaries (V_min, SOC_max, etc.) are satisfied
    reinit::Bool = p.opts.reinit, # reinitialize the initial guess
    verbose::Bool = p.opts.verbose, # print information about the run
    interp_final::Bool = p.opts.interp_final, # interpolate the final points if a boundary is hit
    tstops::AbstractVector = p.opts.tstops, # times the solver explicitly solves for
    tdiscon::AbstractVector = p.opts.tdiscon, # times of known discontinuities in the current function
    interp_bc::Symbol = p.opts.interp_bc, # :interpolate or :extrapolate
    save_start::Bool = p.opts.save_start, # warm-start for the initial guess
    V_max::Number = p.bounds.V_max,
    V_min::Number = p.bounds.V_min,
    SOC_max::Number = p.bounds.SOC_max,
    SOC_min::Number = p.bounds.SOC_min,
    T_max::Number = p.bounds.T_max,
    c_s_n_max::Number = p.bounds.c_s_n_max,
    I_max::Number = p.bounds.I_max,
    I_min::Number = p.bounds.I_min,
    ) where {
        T1<:param,
        T2<:Union{Number,AbstractVector,Nothing},
        R1<:Union{model_output,Vector{Float64}},
        R5<:Union{Tuple,Symbol}
    }
        
    # Force the outputs into a Tuple
    if (R5 === Symbol) outputs = (outputs,) end
        
    # Check if the outputs are the same as in the cache
    if outputs === p.opts.var_keep.results
        var_keep = p.opts.var_keep
    else
        # Create a new struct that matches the specified outputs
        p.opts.outputs = outputs
        p.opts.var_keep = var_keep = model_states_logic(outputs, p.cache.outputs_tot)
    end
        
    # putting opts and bounds into a structure
    t0 = isempty(model) || model isa Array{Float64,1} ? 0.0 : (@inbounds model.t[end])
    run = get_run(I, V, P, res, t0, tspan)
    
    opts = options_model(outputs, Float64(SOC), abstol, reltol, abstol_init, reltol_init, maxiters, check_bounds, reinit, verbose, interp_final, tstops, tdiscon, interp_bc, save_start, var_keep)
    bounds = boundary_stop_conditions(V_max, V_min, SOC_max, SOC_min, T_max, c_s_n_max, I_max, I_min)
    
    # getting the initial conditions and run setup
    int, method_funcs, model = initialize_model!(model, p, run, bounds, opts)

    if !within_bounds(run)
        if verbose @warn "Instantly hit simulation stop conditions: $(run.info.exit_reason)" end
        instant_hit_bounds(model, opts)
        return model
    end

    if verbose println("\n$(run)") end
    
    solve!(model, int, run, p, bounds, opts, method_funcs)
    
    exit_simulation!(p, model, run, bounds, int, opts)

    # if you want to interpolate the results
    if T2 <: AbstractVector model = model(tspan, interp_bc=opts.interp_bc) end

    if verbose println("\n$(model)\n") end

    return model
end
@inline run_model!(_model, x...; model::Nothing=nothing, kw...) = run_model(x...; model=_model, kw...)

@inline within_bounds(run::AbstractRun) = run.info.flag === -1

struct run_container{T1<:AbstractJacobian,T2<:AbstractRun,T3<:Function}
    p::param{T1}
    funcs::functions_model{T1}
    run::T2
    residuals!::T3
    Jacobian!::T1
    θ_tot::Vector{Float64}
end
@inline function initialize_model!(model::model_struct, p::param, run::T, bounds::boundary_stop_conditions, opts::options_model) where {T<:AbstractRun,model_struct<:Union{model_output,Vector{Float64}}}
    if !haskey(p.method_functions,run)
        get_method_funcs!(p,run)
        method_funcs = p.method_functions(run)
    end
    method_funcs = p.method_functions(run)
    θ_tot = method_funcs.J_full.θ_tot
    update_θ!(θ_tot, method_funcs.J_full.θ_keys, p.θ)
    
    cache = p.cache
    
    initial_guess! = p.funcs.initial_guess!
    keep_Y = opts.var_keep.Y
    
    # update the θ_tot vector from the dict p.θ
    check_errors_parameters_runtime(p, opts)
    
    # if this is a new model?
    new_run = model_struct === Array{Float64,1} || isempty(model.results)

    ## initializing the states vector Y and time t
    Y0 = keep_Y ? similar(cache.Y0) : cache.Y0
    YP0 = cache.YP0
    if model_struct === Array{Float64,1}
        Y0 = deepcopy(model)
        model = model_output()
        SOC = calc_SOC((@views @inbounds Y0[p.ind.c_s_avg]), p)
    elseif new_run
        SOC = opts.SOC
        if opts.reinit
            initial_guess!(Y0, SOC, θ_tot, 0.0)
        end
    else # continue from previous simulation
        Y0 .= @inbounds keep_Y ? copy(model.Y[end]) : model.Y[end]
        SOC = @inbounds model.SOC[end]
    end

    @inbounds Y0[p.ind.I] .= run.value .= run.input

    ## getting the DAE integrator function
    initialize_states!(p,Y0,YP0,run,opts,method_funcs,SOC)
    
    int = retrieve_integrator(run,p,method_funcs,Y0,YP0,opts)
    
    set_vars!(model, p, Y0, YP0, int.t, run, opts; init_all=new_run)
    set_var!(model.Y, new_run || keep_Y, Y0)
    
    check_simulation_stop!(model, int, run, p, bounds, opts)
    return int, method_funcs, model
end

@inline function retrieve_integrator(run, p::param, method_funcs::T, Y0, YP0, opts::options_model) where T<:Jac_and_res
    """
    If the model has previously been evaluated for a constant run simulation, you can reuse
    the integrator function with its cache instead of creating a new one
    """
    
    if isempty(method_funcs.int)
        int = create_integrator(run, p, method_funcs, Y0, YP0, opts)
    else
        # reuse the integrator cache
        int = @inbounds method_funcs.int[1]

        # reinitialize at t = 0 with new Y0/YP0 and tolerances
        Sundials.IDASStolerances(int.mem, opts.reltol, opts.abstol)
        Sundials.IDAReInit(int.mem, 0.0, Y0, YP0)
        int.t = 0.0

        postfix_integrator!(int, run, opts)
    end

    return int
end
@inline retrieve_integrator(run::run_function, x...) = create_integrator(run, x...)

@inline function create_integrator(run::T, p::param, method_funcs::Q, Y0, YP0, opts::options_model) where {T<:AbstractRun,Q<:Jac_and_res}
    R_full = method_funcs.R_full
    J_full = method_funcs.J_full
    DAEfunc = DAEFunction(
        (res,YP,Y,extra,t) -> R_full(res,t,Y,YP,p,run);
        jac = (J,YP,Y,extra,γ,t) -> J_full(J,t,Y,YP,γ,p,run),
        jac_prototype = J_full.sp,
    )

    prob = DAEProblem(DAEfunc, YP0, Y0, (0.0, run.tf), J_full.θ_tot, differential_vars=p.cache.id)

    int = DiffEqBase.init(prob, Sundials.IDA(linear_solver=:KLU), tstops=Float64[], abstol=opts.abstol, reltol=opts.reltol, save_everystep=false, save_start=false, verbose=false)

    postfix_integrator!(int, run, opts)

    push!(method_funcs.int, int)

    return int
end
@inline function postfix_integrator!(int::Sundials.IDAIntegrator, run::AbstractRun, opts::options_model)
    tstops = sort!(Float64[
        opts.tstops
        opts.tdiscon .- opts.reltol/2
        run.tf
        ])
    
    # the model can fail is tstops includes 0
    if (@inbounds iszero(tstops[1])) deleteat!(tstops, 1) end
    int.opts.tstops.valtree = tstops

    return nothing
end

@inline function solve!(model,int::R1,run::R2,p::R3,bounds,opts::R4,method_funcs) where {R1<:Sundials.IDAIntegrator,R2<:AbstractRun,R3<:param,R4<:options_model}
    keep_Y = opts.var_keep.Y
    Y = int.u.v
    YP = int.du.v
    
    iter = 1
    status = within_bounds(run)
    @inbounds while status
        step!(int)
        iter += 1
        t = int.t
        
        set_vars!(model, p, Y, YP, t, run, opts)
        
        check_simulation_stop!(model, int, run, p, bounds, opts)
        
        status = check_solve(run, model, int, p, bounds, opts, method_funcs, keep_Y, iter, Y, t)
    end
    
    run.info.iterations = iter
    return nothing
end

# residuals for DAE with a run function
@inline function f!(res::R1, du::R1, u::R1, θ_tot::R1, t::E, container::run_container{T,<:AbstractRun}) where {T<:AbstractJacobian,E<:Float64,R1<:Vector{E}}
    p = container.p
    run = container.run
    
    container.residuals!(res,t,u,du,θ_tot)

    scalar_residual!(res,t,u,du,p,run)

    return nothing
end

# Jacobian for DAE with a run function
@inline function g!(::S, du::R1, u::R1, θ_tot::R1, γ::E, t::E, container::run_container{T,<:AbstractRun}) where {T<:jacobian_symbolic,E<:Float64,R1<:Vector{E},S<:SparseMatrixCSC{E,Int64}}
    container.Jacobian!(t,u,du,γ,θ_tot)
    return nothing
end

@inline function exit_simulation!(p::R1, model::R2, run::R3, bounds::R4, int::R5, opts::R6) where {R1<:param,R2<:model_output,R3<:AbstractRun,R4<:boundary_stop_conditions,R5<:Sundials.IDAIntegrator,R6<:options_model}
    # if a stop condition (besides t = tf) was reached
    if opts.interp_final && !(run.info.flag === 0)
        interp_final_points!(p, model, run, bounds, int, opts)
    else
        set_var!(model.Y, opts.var_keep.Y, int.u.v)
    end

    if isempty(model.results)
        run_index = 1:run.info.iterations
    else
        iterations_start = sum(result.info.iterations for result in model.results)
        run_index = (1:run.info.iterations) .+ iterations_start
    end

    tspan = (run.t0, @inbounds model.t[end])

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
    )

    push!(model.results, results)

    return nothing
end

@views @inbounds @inline function interp_final_points!(p::R1, model::R2, run::R3, bounds::R4, int::R5, opts::R6) where {R1<:param,R2<:model_output,R3<:AbstractRun,R4<:boundary_stop_conditions,R5<:Sundials.IDAIntegrator,R6<:options_model}
    YP = opts.var_keep.YP ? model.YP[end] : Float64[]
    t = bounds.t_final_interp_frac*(int.t - int.tprev) + int.tprev
    
    set_var!(model.Y,  opts.var_keep.Y, bounds.t_final_interp_frac.*(int.u.v .- model.Y[end]) .+ model.Y[end])
    set_vars!(model, p, model.Y[end], YP, t, run, opts; modify! = set_var_last!)
    
    return nothing
end

@inline function save_start_init!(Y0::Vector{Float64}, run::AbstractRun, p::param, SOC::Float64)
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
@inline function initialize_states!(p::param{T}, Y0::R1, YP0::R1, run::AbstractRun, opts::options_model, method_funcs::Jac_and_res, SOC::Float64) where {T<:AbstractJacobian,R1<:Vector{Float64}}
    if opts.save_start
        key, key_exists = save_start_init!(Y0, run, p, SOC)
        
        newtons_method!(p,Y0,YP0,run,opts,method_funcs.R_alg,method_funcs.R_diff,method_funcs.J_alg)
        
        if !key_exists
            p.cache.save_start_dict[key] = @inbounds Y0[(1:p.N.alg) .+ p.N.diff]
        end
    else
        newtons_method!(p,Y0,YP0,run,opts,method_funcs.R_alg,method_funcs.R_diff,method_funcs.J_alg)
    end
    
    return nothing
end

@inline function newtons_method!(p::param,Y::R1,YP::R1,run,opts::options_model,R_alg::T1,R_diff::T2,J_alg::T3; itermax::Int64=1000
    ) where {R1<:Vector{Float64},T1<:residual_combined,T2<:residual_combined,T3<:jacobian_combined}

    res   = p.cache.res
    Y_old = p.cache.Y_alg
    Y_new = @views @inbounds Y[p.N.diff+1:end]
    YP   .= 0.0
    J     = J_alg.sp

    t = γ = 0.0
    # starting loop for Newton's method
    @inbounds for iter in 1:itermax
        # updating Y and J
        R_alg(res,t,Y,YP,p,run)
        J_alg(t,Y,YP,γ,p,run)

        Y_old .= Y_new
        Y_new .-= J\res

        if norm(Y_old .- Y_new) < opts.reltol_init || maximum(abs, res) < opts.abstol_init
            break
        elseif iter === itermax
            error("Could not initialize DAE in $itermax iterations.")
        end
    end
    # calculate the differential equations for YP0
    R_diff(YP,t,Y,YP,p,run)

    return nothing
end

@inline function run_determination(method::AbstractMethod,input::Any,t0::Float64,tspan::Q) where {Q}
    value = [0.0]
    tf = Q === Nothing ? 1e6 : (@inbounds Float64(tspan[end]))

    return run_constant(input, value, method, t0, tf, run_info())
end
@inline function run_determination(method::AbstractMethod,func::Function,t0::Float64, tspan::Q) where {Q}
    value = [0.0]
    tf = Q === Nothing ? 1e6 : (@inbounds Float64(tspan[end]))

    return run_function(func, value, method, t0, tf, run_info())
end
@inline function run_determination(::method_res,func::Function,t0::Float64, tspan::Q) where {Q}
    value = [0.0]
    tf = Q === Nothing ? 1e6 : (@inbounds Float64(tspan[end]))

    return run_residual(func, value, t0, tf, run_info())
end