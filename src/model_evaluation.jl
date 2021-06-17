@inline function run_model(
    p::T1, # initial parameters file
    tspan::T2 = nothing; # a single number (length of the experiment) or a vector (interpolated points)
    model::R1 = model_output(), # using a new model or continuing from previous simulation?
    I = nothing, # constant current C-rate. also accepts :rest
    V = nothing, # constant voltage. also accepts :hold if continuing simulation
    P = nothing, # constant power.   also accepts :hold if continuing simulation
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
    warm_start::Bool = p.opts.warm_start, # warm-start for the initial guess
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
    method_struct = get_method(I, V, P)
    opts = options_model(outputs, Float64(SOC), abstol, reltol, abstol_init, reltol_init, maxiters, check_bounds, reinit, verbose, interp_final, tstops, tdiscon, interp_bc, warm_start, var_keep)
    bounds = boundary_stop_conditions(V_max, V_min, SOC_max, SOC_min, T_max, c_s_n_max, I_max, I_min)
    
    # getting the initial conditions and run setup
    int, run, container, model = initialize_model!(model, p, tspan, method_struct, bounds, opts)

    if !within_bounds(run)
        if verbose @warn "Instantly hit simulation stop conditions: $(run.info.exit_reason)" end
        instant_hit_bounds(model, opts)
        return model
    end

    if verbose println("\n$(run)") end
    
    solve!(model, int, run, p, bounds, opts, container)
    
    exit_simulation!(p, model, run, bounds, int, opts)

    # if you want to interpolate the results
    if T2 <: AbstractVector model = model(tspan, interp_bc=opts.interp_bc) end

    if verbose println("\n$(model)\n") end

    return model
end
@inline run_model!(_model, x...; model::Nothing=nothing, kw...) = run_model(x...; model=_model, kw...)

@inline within_bounds(run::AbstractRun) = run.info.flag === -1

struct run_container{T1<:AbstractJacobian,T2<:AbstractRun}
    p::param{T1}
    funcs::functions_model{T1}
    run::T2
    residuals!::Function
    Jacobian!::T1
    θ_tot::Vector{Float64}
end
@inline function initialize_model!(model::R1, p::param, tspan::T1, method_struct::method_and_value{method,T}, bounds::boundary_stop_conditions, opts::options_model) where {T1<:Union{Number,AbstractArray,Nothing},R1<:Union{model_output,Vector{Float64}},method<:AbstractMethod,T}
    cache = p.cache
    
    method_sym = method_symbol(method)
    funcs = p.funcs[method_sym]
    θ_tot = cache.θ_tot[method_sym]
    
    # update the θ_tot vector from the dict p.θ
    funcs.update_θ!(θ_tot, p.θ)
    check_errors_parameters_runtime(p, opts, tspan)
    
    # if this is a new model?
    new_run = R1 === Array{Float64,1} || isempty(model.results)

    ## initializing the states vector Y and time t    
    Y0 = cache.Y0
    YP0 = cache.YP0
    if R1 === Array{Float64,1}
        Y0 = deepcopy(model)
        model = model_output()
        SOC = calc_SOC((@views @inbounds Y0[p.ind.c_s_avg]), p)
        t0 = 0.0
    elseif new_run
        SOC = opts.SOC
        if opts.reinit
            funcs.initial_guess!(Y0, SOC, θ_tot, 0.0)
        end
        t0 = 0.0

    else # continue from previous simulation
        Y0  .= @views @inbounds copy(model.Y[end])
        SOC = @inbounds model.SOC[end]
        t0 = @inbounds model.t[end]
    end

    run = run_determination(method_struct, p, model, t0, tspan, Y0)
    @inbounds Y0[p.ind.I] .= value(run)

    container = run_container(p, funcs, run, funcs.f!, funcs.J_y!, θ_tot)

    ## getting the DAE integrator function
    initialize_states!(p, Y0, YP0, run, opts, container, SOC)
    
    int = retrieve_integrator(run, p, container, Y0, YP0, opts)
    
    set_vars!(model, p, Y0, YP0, int.t, run, opts; init_all=new_run)
    set_var!(model.Y, new_run, Y0)
    
    check_simulation_stop!(model, int, run, p, bounds, opts)
    
    return int, run, container, model
end

@inline function retrieve_integrator(run::run_constant, p::R1, container::R2, Y0::R3, YP0::R3, opts::options_model) where {T1<:AbstractJacobian,T2<:AbstractRun,R1<:param{T1},R2<:run_container{T1,T2},R3<:Vector{Float64}}
    """
    If the model has previously been evaluated for a constant run simulation, you can reuse
    the integrator function with its cache instead of creating a new one
    """
    
    if isempty(container.funcs.int)
        int = create_integrator(run, p, container, Y0, YP0, opts)
        
        # so the integrator is only created once
        push!(container.funcs.int, int)
    else
        # reuse the integrator cache
        int = @inbounds container.funcs.int[1]

        # reinitialize at t = 0 with new Y0/YP0 and tolerances
        Sundials.IDASStolerances(int.mem, opts.reltol, opts.abstol)
        Sundials.IDAReInit(int.mem, 0.0, Y0, YP0)
        int.t = 0.0

        postfix_integrator!(int, run, opts)
    end

    return int
end
@inline retrieve_integrator(run::run_function, x...) = create_integrator(run, x...)

@inline function create_integrator(run::T2, p::R1, container::R2, Y0::R3, YP0::R3, opts::options_model) where {T1<:AbstractJacobian,T2<:AbstractRun,R1<:param{T1},R2<:run_container{T1,T2},R3<:Vector{Float64}}
    DAEfunc = DAEFunction(
        (res,du,u,p,t) -> f!(res,du,u,p,t,container);
        jac = (J,du,u,p,γ,t) -> g!(J,du,u,p,γ,t,container),
        jac_prototype = container.Jacobian!.sp
    )

    prob =  DAEProblem(DAEfunc, YP0, Y0, (0.0, run.tf), container.θ_tot, differential_vars=p.cache.id)

    int = DiffEqBase.init(prob, Sundials.IDA(linear_solver=:KLU), tstops=Float64[], abstol=opts.abstol, reltol=opts.reltol, save_everystep=false, save_start=false, verbose=false)

    postfix_integrator!(int, run, opts)

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

@inline function solve!(model::R1, int::R2, run::run_constant, p::R3, bounds::R4, opts::R5, container::R6) where {R1<:model_output, R2<:Sundials.IDAIntegrator,R3<:param,R4<:boundary_stop_conditions,R5<:options_model, R6<:run_container}
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
        
        if t === int.tprev
            error("Model failed to converge at t = $(t)")
        elseif iter === opts.maxiters
            error("Reached max iterations of $(opts.maxiters) at t = $(t)")
        elseif within_bounds(run)
            # update Y only after checking the stop conditions. this is done to store a copy of the
            # previous model run in case any back-interpolation is needed
            set_var!(model.Y, keep_Y, keep_Y ? copy(Y) : Y)
        else # no errors and run.info.flag ≠ -1
            status = false
        end
    end
    
    run.info.iterations = iter
    return nothing
end

@inline function solve!(model::R1, int::R2, run::run_function, p::R3, bounds::R4, opts::R5, container::R6) where {R1<:model_output, R2<:Sundials.IDAIntegrator,R3<:param,R4<:boundary_stop_conditions,R5<:options_model, R6<:run_container}
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
        
        if iter === opts.maxiters
            error("Reached max iterations of $(opts.maxiters) at t = $(int.t)")
        elseif within_bounds(run)
            # update Y only after checking the stop conditions. this is done to store a copy of the
            # previous model run in case any back-interpolation is needed
            set_var!(model.Y, keep_Y, keep_Y ? copy(Y) : Y)
            
            # check to see if the run needs to be reinitialized
            if t - int.tprev < 1e-3opts.reltol
                check_reinitialization!(model, int, run, p, bounds, opts, container)
            end
        else # no errors and run.info.flag ≠ -1
            status = false
        end
    end
    
    run.info.iterations = iter
    return nothing
end

# residuals for DAE with a constant run value
@inline function f!(res::R1, du::R1, u::R1, θ_tot::R1, t::E, container::run_container{T,<:run_constant}) where {T<:AbstractJacobian,E<:Float64,R1<:Vector{E}}
    p = container.p
    run = container.run
    
    container.residuals!(res, u, du, θ_tot)

    fix_res!(res, u, p, run)
    
    return nothing
end
# residuals for DAE with a run function
@inline function f!(res::R1, du::R1, u::R1, θ_tot::R1, t::E, container::run_container{T,<:run_function}) where {T<:AbstractJacobian,E<:Float64,R1<:Vector{E}}
    p = container.p
    run = container.run

    @inbounds run.value .= u[p.ind.I] .= run.func(u, p, t)::Float64
    
    container.residuals!(res, u, du, θ_tot)

    fix_res!(res, u, p, run)

    return nothing
end

# Jacobian for DAE with a constant run value
@inline function g!(J::S, du::R1, u::R1, θ_tot::R1, γ::E, t::E, container::run_container{T,<:run_constant}) where {T<:AbstractJacobian,E<:Float64,R1<:Vector{E},S<:SparseMatrixCSC{E,Int64}}
    p = container.p
    
    container.Jacobian!(J, u, θ_tot)
    
    @inbounds for i in 1:p.N.diff
        @inbounds J[i,i] -= γ
    end

    return nothing
end

# Jacobian for DAE with a run function
@inline function g!(J::S, du::R1, u::R1, θ_tot::R1, γ::E, t::E, container::run_container{T,<:run_function}) where {T<:AbstractJacobian,E<:Float64,R1<:Vector{E},S<:SparseMatrixCSC{E,Int64}}
    p = container.p
    run = container.run
    
    @inbounds run.value .= u[p.ind.I] .= run.func(u, p, t)::Float64

    container.Jacobian!(J, u, θ_tot)
    
    @inbounds for i in 1:p.N.diff
        @inbounds J[i,i] -= γ
    end

    return nothing
end

@inline fix_res!(::W, ::W, ::param{T}, ::AbstractRun{Q}; kw...) where {Q<:AbstractMethod,T<:jacobian_symbolic,E<:Float64,W<:Vector{E}} = nothing
@inline function fix_res!(res::W, u::W, p::param{T}, run::AbstractRun{Q}; offset::Int=0) where {Q<:run_voltage,T<:jacobian_symbolic,E<:Float64,W<:Vector{E}}
    @inbounds res[p.ind.I[1]+offset] = calc_V(u, p, run, p.ind.Φ_s.+offset) - value(run)
    return nothing
end

@inline function fix_res!(res::W, u::W, p::param{T}, run::AbstractRun{Q}; offset::Int=0) where {Q<:run_voltage,T<:jacobian_AD,E<:Float64,W<:Vector{E}}
    @inbounds res[p.ind.I[1]+offset] = calc_V(u, p, run, p.ind.Φ_s.+offset) - value(run)
    return nothing
end
@inline function fix_res!(res::W, u::W, p::param{T}, run::AbstractRun{Q}; offset::Int=0) where {Q<:Union{run_current,run_power},T<:jacobian_AD,E<:Float64,W<:Vector{E}}
    @inbounds res[p.ind.I[1]+offset] = 0.0
    return nothing
end

@inline function f_IC!(res::T1, Y_alg::T1, Y_diff::T1, p::param, run::AbstractRun, f!::Function, θ_tot::T1) where {T1<:Vector{Float64}}
    f!(res, Y_alg, Y_diff, θ_tot)

    fix_res!(res, Y_alg, p, run, offset=-p.N.diff)
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

@inline function warm_start_init!(Y0::Vector{Float64}, run::AbstractRun, p::param, SOC::Float64)
    key = warm_start_info(
        run.method,
        round(SOC, digits=4),
        round(value(run), digits=4),
    )

    warm_start_dict = p.cache.warm_start_dict
    
    key_exists = key ∈ keys(warm_start_dict)
    if key_exists
        @inbounds Y0[(1:p.N.alg) .+ p.N.diff] .= warm_start_dict[key]
    end
    
    return key, key_exists
end
@inline function initialize_states!(p::param{T}, Y0::R1, YP0::R1, run::AbstractRun, opts::options_model, container::run_container{T}, SOC::Float64) where {T<:AbstractJacobian,R1<:Vector{Float64}}
    if opts.warm_start
        key, key_exists = warm_start_init!(Y0, run, p, SOC)
        
        newtons_method!(p, Y0, YP0, run, opts, container)
        
        if !key_exists
            p.cache.warm_start_dict[key] = @inbounds Y0[(1:p.N.alg) .+ p.N.diff]
        end
    else
        newtons_method!(p, Y0, YP0, run, opts, container)
    end
    
    return nothing
end

@inline function newtons_method!(p::param{T}, Y0::R1, YP0::R1, run::AbstractRun, opts::options_model, container::run_container{T}) where {T<:AbstractJacobian,R1<:Vector{Float64}}
    itermax = 1000
    funcs = container.funcs
    IC = funcs.initial_conditions
    
    f! = IC.f_alg!
    g! = IC.J_y_alg!
    θ_tot = container.θ_tot

    @inbounds @views IC.Y0_alg  .= Y0[(1:p.N.alg) .+ p.N.diff]
    @inbounds @views IC.Y0_diff .= Y0[(1:p.N.diff)]

    # retrieving variables from cache
    Y = IC.Y0_alg
    Y_diff = IC.Y0_diff
    Y_old = IC.Y0_alg_prev
    res = IC.res
    J = IC.J_y_alg!.sp
    
    # starting loop for Newton's method
    iter = 0
    @inbounds while true
        # updating Y and J
        f_IC!(res, Y, Y_diff, p, run, f!, θ_tot)
        g!(J, Y, Y_diff, θ_tot)

        Y_old .= Y
        Y .-= J\res

        iter += 1
        if norm(Y_old .- Y) < opts.reltol_init || maximum(abs, res) < opts.abstol_init
            @inbounds Y0[(1:p.N.alg) .+ p.N.diff] .= Y
            break
        elseif iter === itermax
            error("Could not initialize DAE in $itermax iterations.")
        end
    end
    # set YP0 to zeros
    fill!(YP0, 0)
    # calculate the differential equations for YP0
    IC.f_diff!(YP0, Y0, YP0, θ_tot)
    
    return nothing
end

@inline function run_determination(method_struct::method_and_value{<:run_current,Y}, p::param, model::R1, t0::Float64, tspan::T1, Y0::R2) where {Y<:Union{Number,Symbol},T1<:Union{Number,AbstractVector,Nothing},R1<:model_output,R2<:Vector{Float64}}
    method = method_struct.method
    I      = method_struct.value

    if Y === Symbol
        if     I === :rest
            I = 0.0
        elseif I === :hold
            I = @inbounds model.I[end]
        else
            error("I can only be a `:rest`, `:hold`, or a number")
        end
    else
        I = Float64(I)
    end

    value = I
    tf = T1 === Nothing ? 2.0*(3600.0/abs(I)) : (@inbounds Float64(tspan[end]))

    return run_constant(value, method, t0, tf, run_info())
end
@inline function run_determination(method_struct::method_and_value{<:run_current,Y}, p::param, model::R1, t0::Float64, tspan::T1, Y0::R2) where {Y<:Function,T1<:Union{Number,AbstractVector,Nothing},R1<:model_output,R2<:Vector{Float64}}
    method = method_struct.method
    I      = method_struct.value

    func = I
    tf = T1 === Nothing ? 1e10 : (@inbounds Float64(tspan[end]))
    value = [0.0]

    run = run_function(func, value, method, t0, tf, run_info())
    value .= func(Y0, p, t0)::Float64
    return run
end
@inline function run_determination(method_struct::method_and_value{<:run_voltage,Y}, p::param, model::R1, t0::Float64, tspan::T1, Y0::R2) where {Y<:Union{Number,Symbol},T1<:Union{Number,AbstractVector,Nothing},R1<:model_output,R2<:Vector{Float64}}
    method = method_struct.method
    V      = method_struct.value

    if Y === Symbol
        if isempty(model.results)
            error("`V = :hold` can only be used with a previous model")
        end
        V = @inbounds model.V[end]
    else
        V = Float64(V)
    end
    value = V
    tf = @inbounds T1 === Nothing ? 1e10 : Float64(tspan[end])

    return run_constant(value, method, t0, tf, run_info())
end
@inline function run_determination(method_struct::method_and_value{<:run_power,Y}, p::param, model::R1, t0::Float64, tspan::T1, Y0::R2) where {Y<:Union{Number,Symbol},T1<:Union{Number,AbstractVector,Nothing},R1<:model_output,R2<:Vector{Float64}}
    method = method_struct.method
    P      = method_struct.value

    if Y === Symbol
        if isempty(model.results)
            error("`P = :hold` can only be used with a previous model")
        else
            P = @inbounds model.P[end]
        end
    else
        P = Float64(P)
    end
    value = P
    tf = T1 === Nothing ? 1e10 : (@inbounds Float64(tspan[end]))

    return run_constant(value, method, t0, tf, run_info())
end
@inline function run_determination(method_struct::method_and_value{<:run_power,Y}, p::param, model::R1, t0::Float64, tspan::T1, Y0::R2) where {Y<:Function,T1<:Union{Number,AbstractVector,Nothing},R1<:model_output,R2<:Vector{Float64}}
    method = method_struct.method
    P      = method_struct.value

    func = P
    tf = T1 === Nothing ? 1e10 : (@inbounds Float64(tspan[end]))
    value = [0.0]
    
    run = run_function(func, value, method, t0, tf, run_info())
    value .= func(Y0, p, t0)::Float64
    return run
end
