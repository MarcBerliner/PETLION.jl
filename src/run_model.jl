@inline function run_model(
    p::T1, # initial parameters file
    tf::T2 = nothing; # a single number (length of the experiment) or a vector (interpolated points)
    model::R1 = model_output(), # using a new model or continuing from previous simulation?
    I   = nothing, # constant current C-rate. also accepts :rest
    V   = nothing, # constant voltage. also accepts :hold if continuing simulation
    P   = nothing, # constant power.   also accepts :hold if continuing simulation
    SOC::Number     = p.opts.SOC, # initial SOC of the simulation. only valid if not continuing simulation
    outputs::R2     = p.opts.outputs, # model output states
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
    ) where {
        T1<:param,
        T2<:Union{Number,AbstractVector,Nothing},
        R1<:Union{model_output,Vector{Float64}},
        R2<:Union{Tuple,Symbol}
    }
    
    # Force the outputs into a Tuple
    if (R2 === Symbol)
        outputs = (outputs,)
    end
        
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
    run = get_run(I,V,P,t0,tf,p,model)
    
    # putting opts and bounds into a structure. 
    opts = options_model_immutable(outputs, Float64(SOC), abstol, reltol, abstol_init, reltol_init, maxiters, check_bounds, reinit, verbose, interp_final, tstops, tdiscon, interp_bc, save_start, var_keep, stop_function, calc_integrator)
    bounds = boundary_stop_conditions(V_max, V_min, SOC_max, SOC_min, T_max, c_s_n_max, I_max, I_min, η_plating_min, c_e_min, dfilm_max)
    
    # getting the initial conditions and run setup
    int, funcs, model = initialize_model!(model, p, run, bounds, opts)

    if !within_bounds(run)
        if verbose @warn "Instantly hit simulation stop conditions: $(run.info.exit_reason)" end
        exit_simulation!(p, model, run, bounds, int, opts; cancel_interp=true)

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
@inline run_model!(_model,x...; overwrite_model::Bool=false, model::Nothing=nothing, kw...) = run_model(x...; model=overwrite_model ? deepcopy(_model) : _model, kw...)