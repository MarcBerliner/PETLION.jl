@inline function run_model(
    p::T1, # initial parameters file
    tf::T2 = nothing; # a single number (length of the experiment) or a vector (interpolated points)
    model::R1 = model_output(), # using a new model or continuing from previous simulation?
    I   = nothing, # constant current C-rate. also accepts :rest
    V   = nothing, # constant voltage. also accepts :hold if continuing simulation
    P   = nothing, # constant power.   also accepts :hold if continuing simulation
    η_p = nothing, # plating overpotential
    res = nothing, # custom residual function
    res_I_guess = 1.0, # initial guess for I when using the residuals function
    dT         = nothing, # temperature derivative
    dc_s_p_max = nothing,
    dc_s_p_min = nothing,
    dc_s_n_max = nothing,
    dc_s_n_min = nothing,
    dc_e_max   = nothing,
    dc_e_min   = nothing,
    SOC::Number   = p.opts.SOC, # initial SOC of the simulation. only valid if not continuing simulation
    outputs::R2   = p.opts.outputs, # model output states
    abstol        = p.opts.abstol, # absolute tolerance in DAE solve
    reltol        = p.opts.reltol, # relative tolerance in DAE solve
    abstol_init   = abstol, # absolute tolerance in initialization
    reltol_init   = reltol, # relative tolerance in initialization
    maxiters      = p.opts.maxiters, # maximum solver iterations
    check_bounds  = p.opts.check_bounds, # check if the boundaries (V_min, SOC_max, etc.) are satisfied
    reinit        = p.opts.reinit, # reinitialize the initial guess
    verbose       = p.opts.verbose, # print information about the run
    interp_final  = p.opts.interp_final, # interpolate the final points if a boundary is hit
    tstops        = p.opts.tstops, # times the solver explicitly solves for
    tdiscon       = p.opts.tdiscon, # times of known discontinuities in the current function
    interp_bc     = p.opts.interp_bc, # :interpolate or :extrapolate
    save_start    = p.opts.save_start, # warm-start for the initial guess
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
    ) where {
        T1<:param,
        T2<:Union{Number,AbstractVector,Nothing},
        R1<:Union{model_output,Vector{Float64}},
        R2<:Union{Tuple,Symbol}
    }
    
    # Force the outputs into a Tuple
    if (R2 === Symbol) outputs = (outputs,) end
        
    # Check if the outputs are the same as in the cache
    if outputs === p.opts.var_keep.results
        var_keep = p.opts.var_keep
    else
        # Create a new struct that matches the specified outputs
        p.opts.outputs = outputs
        p.opts.var_keep = var_keep = model_states_logic(outputs, p.cache.outputs_tot)
    end
        
    # `nextfloat` ensures that there is always a unique point for interpolation
    t0 = isempty(model) || model isa Array{Float64,1} ? 0.0 : (@inbounds nextfloat(model.t[end]))
    
    # identifying the run type
    run = get_run(I,V,P,η_p,res,dT,dc_s_p_max,dc_s_p_min,dc_s_n_max,dc_s_n_min,dc_e_max,dc_e_min,t0,tf,p,model)
    
    # putting opts and bounds into a structure. 
    opts = options_model(outputs, Float64(SOC), abstol, reltol, abstol_init, reltol_init, maxiters, check_bounds, reinit, verbose, interp_final, tstops, tdiscon, interp_bc, save_start, var_keep)
    bounds = boundary_stop_conditions(V_max, V_min, SOC_max, SOC_min, T_max, c_s_n_max, I_max, I_min, η_plating_min, c_e_min)
    
    # getting the initial conditions and run setup
    int, funcs, model = initialize_model!(model, p, run, bounds, opts, res_I_guess)

    if !within_bounds(run)
        if verbose @warn "Instantly hit simulation stop conditions: $(run.info.exit_reason)" end
        instant_hit_bounds(model, opts)
        return model
    end

    if verbose println("\n$(run)") end
    
    solve!(model, int, run, p, bounds, opts, funcs)
    
    exit_simulation!(p, model, run, bounds, int, opts)

    # if you want to interpolate the results
    if T2 <: AbstractVector model = model(tf, interp_bc=opts.interp_bc) end

    if verbose println("\n$(model)\n") end

    return model
end
@inline run_model!(_model, x...; model::Nothing=nothing, kw...) = run_model(x...; model=_model, kw...)

@inline function get_run(
    I::current,
    V::voltage,
    P::power,
    η_p::plating_overpotential,
    res::residual,
    dT::d_temperature,
    dc_s_p_max::d_solid_surf_conc_cathode_max,
    dc_s_p_min::d_solid_surf_conc_cathode_min,
    dc_s_n_max::d_solid_surf_conc_anode_max,
    dc_s_n_min::d_solid_surf_conc_anode_min,
    dc_e_max::d_electrolyte_conc_max,
    dc_e_min::d_electrolyte_conc_min,
    t0, tf::Q, p::param, model::model_output) where {
    current  <: Union{Number,Symbol,Function,Nothing},
    voltage  <: Union{Number,Symbol,Function,Nothing},
    power    <: Union{Number,Symbol,Function,Nothing},
    plating_overpotential <: Union{Number,Symbol,Function,Nothing},
    residual                      <: Union{Function,Nothing},
    d_temperature                 <: Union{Number,Symbol,Function,Nothing},
    d_solid_surf_conc_cathode_max <: Union{Number,Symbol,Function,Nothing},
    d_solid_surf_conc_cathode_min <: Union{Number,Symbol,Function,Nothing},
    d_solid_surf_conc_anode_max   <: Union{Number,Symbol,Function,Nothing},
    d_solid_surf_conc_anode_min   <: Union{Number,Symbol,Function,Nothing},
    d_electrolyte_conc_max        <: Union{Number,Symbol,Function,Nothing},
    d_electrolyte_conc_min        <: Union{Number,Symbol,Function,Nothing},
    Q,
    }

    if !( sum(!(method === Nothing) for method in (current,voltage,power,plating_overpotential,residual,d_temperature,d_solid_surf_conc_cathode_max,d_solid_surf_conc_cathode_min,d_solid_surf_conc_anode_max,d_solid_surf_conc_anode_min,d_electrolyte_conc_max,d_electrolyte_conc_min)) === 1 )
        error("Cannot select more than one input")
    end

    if     !(current === Nothing)
        method, input, name = method_I(), I, "I"
    elseif !(voltage === Nothing)
        method, input, name = method_V(), V, "V"
    elseif !(power === Nothing)
        method, input, name = method_P(), P, "P"
    elseif !(plating_overpotential === Nothing)
        method, input, name = method_η_p(), η_p, "η_p"
    elseif !(residual === Nothing)
        p.θ[:_residual_val] = 0.0
        method, input, name = method_res(), res, "res"
    elseif !(d_temperature === Nothing)
        if !(p.numerics.temperature === true) error("Temperature must be enabled when using `dT`.") end
        func = custom_res!(p,dT,constant_temperature,model)
        method, input, name = method_res(), func, "dT"
    elseif !(d_solid_surf_conc_cathode_max === Nothing)
        func = custom_res!(p,dc_s_p_max,dc_s((@views @inbounds argmax(model.Y[end][c_s_indices(p,:p;surf=true)]))),model)
        method, input, name = method_res(), func, "dc_s_p_max"
    elseif !(d_solid_surf_conc_cathode_min === Nothing)
        func = custom_res!(p,dc_s_p_min,dc_s((@views @inbounds argmin(model.Y[end][c_s_indices(p,:p;surf=true)]))),model)
        method, input, name = method_res(), func, "dc_s_p_min"
    elseif !(d_solid_surf_conc_anode_max === Nothing)
        func = custom_res!(p,dc_s_n_max,dc_s((@views @inbounds argmax(model.Y[end][c_s_indices(p,:n;surf=true)]))),model)
        method, input, name = method_res(), func, "dc_s_n_max"
    elseif !(d_solid_surf_conc_anode_min === Nothing)
        func = custom_res!(p,dc_s_n_min,dc_s((@views @inbounds argmin(model.Y[end][c_s_indices(p,:n;surf=true)]))),model)
        method, input, name = method_res(), func, "dc_s_n_min"
    elseif !(d_electrolyte_conc_max === Nothing)
        func = custom_res!(p,dc_e_max,dc_e((@views @inbounds argmax(model.Y[end][p.ind.c_e]))),model)
        method, input, name = method_res(), func, "dc_e_max"
    elseif !(d_electrolyte_conc_min === Nothing)
        func = custom_res!(p,dc_e_min,dc_e((@views @inbounds argmin(model.Y[end][p.ind.c_e]))),model)
        method, input, name = method_res(), func, "dc_e_min"
    else
        error("Method not supported")
    end

    if input isa Function && method != method_res
        input = redefine_func(input)
    end

    value = [0.0]
    tf = (Q === Nothing ? 1e6 : (@inbounds Float64(tf[end])))
    
    run_type = run_determination(method, input)
    run      = run_type(input,value,method,t0,tf,name,run_info())

    return run
end

@inline run_determination(::AbstractMethod,::Any)      = run_constant
@inline run_determination(::AbstractMethod,::Function) = run_function
@inline run_determination(::method_res,::Function)     = run_residual

function custom_res!(p::param,x::T,func_RHS::Q,model::model_output;kw...) where {T<:Function,Q<:Function}
    func = redefine_func(x)
    p.θ[:_residual_val] = 0.0
    return (t,Y,YP,p) -> func_RHS(t,Y,YP,p) - func(t,Y,YP,p)
end
function custom_res!(p::param,x::T,func_RHS::Q,model::model_output;kw...) where {T<:Number,Q<:Function}
    p.θ[:_residual_val] = Float64(x)
    return func_RHS
end
function custom_res!(p::param,x::T,func_RHS::Q,model::model_output;hold_val::Number=0.0,kw...) where {T<:Symbol,Q<:Function}
    if x === :hold
        p.θ[:_residual_val] = hold_val
    else
        error("Mode not supported")
    end
    return func_RHS
end

@inline within_bounds(run::AbstractRun) = run.info.flag === -1

@inline function initialize_model!(model::model_struct, p::param{jac}, run::T, bounds::boundary_stop_conditions, opts::options_model, res_I_guess) where {jac<:AbstractJacobian,method<:AbstractMethod,T<:AbstractRun{method,<:Any},model_struct<:Union{model_output,Vector{Float64}}}
    if !haskey(p.funcs,run)
        get_method_funcs!(p,run)
        funcs = p.funcs(run)
    end
    funcs = p.funcs(run)
    θ_tot = funcs.J_full.θ_tot
    update_θ!(θ_tot, funcs.J_full.θ_keys, p.θ)
    if jac === jacobian_AD; (@views @inbounds (p.cache.θ_tot .= θ_tot[1:length(p.cache.θ_tot)])) end
    
    cache = p.cache
    
    initial_guess! = p.funcs.initial_guess!
    keep_Y = opts.var_keep.Y
    
    # update the θ_tot vector from the dict p.θ
    check_errors_parameters_runtime(p, opts)
    
    # if this is a new model?
    new_run = model_struct === Array{Float64,1} || isempty(model)

    ## initializing the states vector Y and time t
    Y0 = keep_Y ? zeros(Float64,length(cache.Y0)) : cache.Y0
    YP0 = cache.YP0
    if model_struct === Array{Float64,1}
        Y0 = deepcopy(model)
        model = model_output()
        SOC = calc_SOC(Y0, p)
    elseif new_run
        SOC = opts.SOC
        if opts.reinit
            initial_guess!(Y0, SOC, θ_tot, 0.0)
        end
    else # continue from previous simulation
        Y0 .= @inbounds keep_Y ? copy(model.Y[end]) : model.Y[end]
        SOC = @inbounds model.SOC[end]
    end

    initial_current!(Y0,YP0,p,run,model,res_I_guess)
    
    ## getting the DAE integrator function
    initialize_states!(p,Y0,YP0,run,opts,funcs,SOC)
    
    int = retrieve_integrator(run,p,funcs,Y0,YP0,opts,new_run)
    
    set_vars!(model, p, Y0, YP0, int.t, run, opts, bounds; init_all=new_run)
    set_var!(model.Y, new_run || keep_Y, Y0)
    
    check_simulation_stop!(model, 0.0, Y0, YP0, run, p, bounds, opts)
    return int, funcs, model
end

@inline function retrieve_integrator(run::T, p::param, funcs::Jac_and_res{<:Sundials.IDAIntegrator}, Y0, YP0, opts::options_model, new_run::Bool) where {method<:AbstractMethod,T<:AbstractRun{method,<:Any}}
    """
    If the model has previously been evaluated for a constant run simulation, you can reuse
    the integrator function with its cache instead of creating a new one
    """
    
    if isempty(funcs.int)
        int = create_integrator(run, p, funcs, Y0, YP0, opts)
    else
        # reuse the integrator cache
        int = @inbounds funcs.int[end]
        @inbounds int.p.value .= run.value

        mem = int.mem
        # reinitialize at t = 0 with new Y0/YP0 and tolerances
        Sundials.IDASStolerances(mem, opts.reltol, opts.abstol)
        Sundials.IDAReInit(mem, 0.0, Y0, YP0)
        int.t = 0.0
    end

    postfix_integrator!(int, run, opts, new_run)

    return int
end

@inline function create_integrator(run::T, p::param, funcs::Q, Y0, YP0, opts::options_model) where {T<:AbstractRun,Q<:Jac_and_res}
    R_full = funcs.R_full
    J_full = funcs.J_full
    DAEfunc = DAEFunction(
        (res,YP,Y,run,t) -> R_full(res,t,Y,YP,p,run);
        jac = (J,YP,Y,run,γ,t) -> J_full(J,t,Y,YP,γ,p,run),
        jac_prototype = J_full.sp,
    )

    prob = DAEProblem(DAEfunc, YP0, Y0, (0.0, run.tf), run, differential_vars=p.cache.id)

    int = DiffEqBase.init(prob, Sundials.IDA(linear_solver=:KLU), tstops=Float64[], abstol=opts.abstol, reltol=opts.reltol, save_everystep=false, save_start=false, verbose=false)

    if isempty(funcs.int)
        push!(funcs.int, int)
    end

    return int
end
@inline function postfix_integrator!(int::Sundials.IDAIntegrator, run::AbstractRun, opts::options_model, new_run::Bool)
    tstops = int.opts.tstops.valtree
    empty!(tstops)
    
    append!(tstops,opts.tstops)
    append!(tstops,opts.tdiscon .- opts.reltol/2)
    push!(tstops,run.tf)
    if !new_run prepend!(tstops, 1.0) end

    sort!(tstops)

    # the model can fail is tstops includes 0
    if (@inbounds iszero(tstops[1])) deleteat!(tstops, 1) end
    return nothing
end

@inline function solve!(model,int::R1,run::R2,p,bounds,opts::R3,funcs) where {R1<:Sundials.IDAIntegrator,R2<:AbstractRun,R3<:options_model}
    keep_Y = opts.var_keep.Y
    Y  = int.u.v
    YP = int.du.v
    
    iter = 1
    status = within_bounds(run)
    @inbounds while status
        step!(int)
        iter += 1
        t = int.t

        set_vars!(model, p, Y, YP, t, run, opts, bounds)
        
        check_simulation_stop!(model, t, Y, YP, run, p, bounds, opts)
        
        status = check_solve(run, model, int, p, bounds, opts, funcs, keep_Y, iter, Y, t)
    end
    
    run.info.iterations = iter
    return nothing
end

@inline function exit_simulation!(p::R1, model::R2, run::R3, bounds::R4, int::R5, opts::R6) where {R1<:param,R2<:model_output,R3<:AbstractRun,R4<:boundary_stop_conditions,R5<:Sundials.IDAIntegrator,R6<:options_model}
    # if a stop condition (besides t = tf) was reached
    if opts.interp_final && !(run.info.flag === 0)
        interp_final_points!(p, model, run, bounds, int, opts)
    else
        set_var!(model.Y, opts.var_keep.Y, opts.var_keep.Y ? copy(int.u.v) : int.u.v)
    end

    iterations_start = isempty(model) ? 0 : (@inbounds @views model.results[end].run_index[end])

    run_index = (1:run.info.iterations) .+ iterations_start

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
        p,
    )

    push!(model.results, results)

    return nothing
end

@views @inbounds @inline function interp_final_points!(p::R1, model::R2, run::R3, bounds::R4, int::R5, opts::R6) where {R1<:param,R2<:model_output,R3<:AbstractRun,R4<:boundary_stop_conditions,R5<:Sundials.IDAIntegrator,R6<:options_model}
    YP = opts.var_keep.YP ? model.YP[end] : Float64[]
    t = bounds.t_final_interp_frac*(int.t - int.tprev) + int.tprev
    
    set_var!(model.Y,  opts.var_keep.Y, bounds.t_final_interp_frac.*(int.u.v .- model.Y[end]) .+ model.Y[end])
    set_vars!(model, p, model.Y[end], YP, t, run, opts, bounds; modify! = set_var_last!)
    
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
@inline function initialize_states!(p::param{T}, Y0::R1, YP0::R1, run::AbstractRun, opts::options_model, funcs::Jac_and_res, SOC::Float64) where {T<:AbstractJacobian,R1<:Vector{Float64}}
    if opts.save_start
        key, key_exists = save_start_init!(Y0, run, p, SOC)
        
        newtons_method!(p,Y0,YP0,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg)
        
        if !key_exists
            p.cache.save_start_dict[key] = @inbounds Y0[(1:p.N.alg) .+ p.N.diff]
        end
    else
        newtons_method!(p,Y0,YP0,run,opts,funcs.R_alg,funcs.R_diff,funcs.J_alg)
    end
    
    return nothing
end

@inline function newtons_method!(p::param,Y::R1,YP::R1,run,opts::options_model,R_alg::T1,R_diff::T2,J_alg::T3;
    itermax::Int64=100, t::Float64=0.0, γ::Float64=0.0
    ) where {R1<:Vector{Float64},T1<:residual_combined,T2<:residual_combined,T3<:jacobian_combined}

    res   = p.cache.res
    Y_old = p.cache.Y_alg
    Y_new = @views @inbounds Y[p.N.diff+1:end]
    YP   .= 0.0
    J     = J_alg.sp

    # starting loop for Newton's method
    @inbounds for iter in 1:itermax
        # updating res, Y, and J
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

"""
Current
"""
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model, res_I_guess) where {method<:method_I,in<:Number}
    input = run.input
    @inbounds run.value .= Y0[end] = input
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:method_I,in<:Symbol}
    input = run.input
    if check_is_hold(input,model)
        @inbounds run.value .= Y0[end] = model.I[end]
    elseif input === :rest
        @inbounds run.value .= Y0[end] = 0.0
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0::Vector{Float64},p,run::run_function{method,func},model, res_I_guess) where {method<:method_I,func<:Function}
    run.value .= Y0[end] = run.func(0.0,Y0,YP0,p)
    return nothing
end

"""
Power
"""
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model, res_I_guess) where {method<:method_P,in<:Number}
    input = run.input
    @inbounds run.value .= Y0[end] = input/(calc_V(Y0,p)*p.θ[:I1C])
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:method_P,in<:Symbol}
    input = run.input
    if check_is_hold(input,model)
        @inbounds run.value .= Y0[end] = model.P[end]
    elseif input === :rest
        @inbounds run.value .= Y0[end] = 0.0
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0::Vector{Float64},p,run::run_function{method,func},model, res_I_guess) where {method<:method_P,func<:Function}
    run.value .= Y0[end] = run.func(0.0,Y0,YP0,p)/(calc_V(Y0,p)*p.θ[:I1C])
    return nothing
end

"""
Voltage and η_plating
"""
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:Union{method_V,method_η_p},in<:Number}
    input = run.input
    @inbounds run.value .= input
    if !isempty(model)
        @inbounds Y0[end] = model.I[end]
    else
        OCV = calc_V(Y0,p)
        @inbounds Y0[end] = input > OCV ? +1.0 : -1.0
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:method_V,in<:Symbol}
    input = run.input
    if check_is_hold(input,model)
        @inbounds run.value .= model.V[end]
        @inbounds Y0[end] = model.I[end]
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},model::model_output, res_I_guess) where {method<:method_η_p,in<:Symbol}
    input = run.input
    if check_is_hold(input,model)
        @inbounds @views run.value .= calc_η_plating(model.Y[end],p)
        @inbounds Y0[end] = model.I[end]
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_function{method,func},model::model_output, res_I_guess) where {method<:Union{method_V,method_η_p},func<:Function}
    @inbounds run.value .= run.func(0.0,Y0,YP0,p)
    if !isempty(model)
        @inbounds Y0[end] = model.I[end]
    else
        OCV = calc_V(Y0,p)
        # Arbitrary guess for the initial current. 
        @inbounds Y0[end] = value(run) > OCV ? +1.0 : -1.0
    end
    return nothing
end

@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_residual{method,func},model::model_output, res_I_guess) where {method<:method_res,func<:Function}
    if !isempty(model)
        @inbounds Y0[end] = model.I[end]
    else
        @inbounds Y0[end] = res_I_guess
    end
    return nothing
end