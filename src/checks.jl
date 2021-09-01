@views @inbounds @inline function check_simulation_stop!(model::R1, t::Float64, Y::R2, run::R3, p::R4, bounds::R5, opts::R6;
    ϵ::Float64 = t > 1 ? 0.0 : opts.reltol,
    ) where {R1<:model_output, R2<:Vector{Float64}, method<:AbstractMethod, R3<:AbstractRun{method},R4<:param,R5<:boundary_stop_conditions,R6<:options_model}
    tf = run.tf
    
    if t ≥ tf
        run.info.flag = 0
        run.info.exit_reason = "Final time reached"
    end

    # continue checking the bounds or return after only evaluating the time
    !opts.check_bounds && (return nothing)
    
    V = model.V[end]
    I = model.I[end]
    if !(method === method_V)
        if (bounds.V_min - V > ϵ) && I < 0
            t_frac = (bounds.V_prev - bounds.V_min)/(bounds.V_prev - V)
            if t_frac < bounds.t_final_interp_frac
                bounds.t_final_interp_frac = t_frac
                run.info.flag = 1
                run.info.exit_reason = "Below minimum voltage limit"
            end
        end

        if (V - bounds.V_max > ϵ) && I > 0
            t_frac = (bounds.V_prev - bounds.V_max)/(bounds.V_prev - V)
            if t_frac < bounds.t_final_interp_frac
                bounds.t_final_interp_frac = t_frac
                run.info.flag = 2
                run.info.exit_reason = "Above maximum voltage limit"
            end
        end
    end
    
    SOC = model.SOC[end]
    if (bounds.SOC_min - SOC > ϵ) && I < 0
        t_frac = (bounds.SOC_prev - bounds.SOC_min)/(bounds.SOC_prev - SOC)
        if t_frac < bounds.t_final_interp_frac
            bounds.t_final_interp_frac = t_frac
            run.info.flag = 3
            run.info.exit_reason = "Below minimum SOC limit"
        end
    end
    if (SOC - bounds.SOC_max > ϵ) && I > 0
        t_frac = (bounds.SOC_prev - bounds.SOC_max)/(bounds.SOC_prev - SOC)
        if t_frac < bounds.t_final_interp_frac
            bounds.t_final_interp_frac = t_frac
            run.info.flag = 4
            run.info.exit_reason = "Above maximum SOC limit"
        end
    end
    
    if p.numerics.temperature
        T = temperature_weighting((@inbounds @views Y[p.ind.T]),p)
        if T - bounds.T_max > ϵ
            t_frac = (bounds.T_prev - bounds.T_max)/(bounds.T_prev - T)
            if t_frac < bounds.t_final_interp_frac
                bounds.t_final_interp_frac = t_frac
                run.info.flag = 5
                run.info.exit_reason = "Above maximum permitted temperature"
            end
        end
    else
        T = -1.0
    end
    
    c_s_avg = Y[p.ind.c_s_avg]
    if !isnan(bounds.c_s_n_max) && I > 0
        if p.numerics.solid_diffusion === :Fickian
            c_s_n_max = maximum(c_s_avg[(p.N.p)*(p.N.r_p)+1:end])
        else
            c_s_n_max = maximum(c_s_avg[(p.N.p)+1:end])
        end
        
        if c_s_n_max - bounds.c_s_n_max*p.θ[:c_max_n] > ϵ
            t_frac = (bounds.c_s_n_prev - bounds.c_s_n_max*(p.θ[:c_max_n]))/(bounds.c_s_n_prev - c_s_n_max)
            if t_frac < bounds.t_final_interp_frac
                bounds.t_final_interp_frac = t_frac
                run.info.flag = 6
                run.info.exit_reason = "Above c_s_n saturation threshold"
            end
        end
    else
        c_s_n_max = -1.0
    end

    if !(method === method_I) && (I - bounds.I_max > ϵ)
        t_frac = (bounds.I_prev - bounds.I_max)/(bounds.I_prev - I)
        if t_frac < bounds.t_final_interp_frac
            bounds.t_final_interp_frac = t_frac
            run.info.flag = 7
            run.info.exit_reason = "Above maximum permitted C-rate"
        end
    end
    if !(method === method_I) && (bounds.I_min - I > ϵ)
        t_frac = (bounds.I_prev - bounds.I_min)/(bounds.I_prev - I)
        if t_frac < bounds.t_final_interp_frac
            bounds.t_final_interp_frac = t_frac
            run.info.flag = 8
            run.info.exit_reason = "Below minimum permitted C-rate"
        end
    end

    η_plating = Y[p.ind.Φ_s.n[1]] - Y[p.ind.Φ_e.n[1]]
    if !isnan(bounds.η_plating_min) && bounds.η_plating_min - η_plating > ϵ
        t_frac = (bounds.η_plating_prev - bounds.η_plating_min)/(bounds.η_plating_prev - η_plating)
        if t_frac < bounds.t_final_interp_frac
            bounds.t_final_interp_frac = t_frac
            run.info.flag = 9
            run.info.exit_reason = "Below minimum permitted η_plating"
        end
    end

    c_e_min = minimum(Y[p.ind.c_e])
    if !isnan(bounds.c_e_min) && bounds.c_e_min - c_e_min > ϵ
        t_frac = (bounds.c_e_min_prev - bounds.c_e_min)/(bounds.c_e_min_prev - c_e_min)
        if t_frac < bounds.t_final_interp_frac
            bounds.t_final_interp_frac = t_frac
            run.info.flag = 9
            run.info.exit_reason = "Below minimum permitted c_e"
        end
    end

    if within_bounds(run)
        bounds.V_prev         = V
        bounds.I_prev         = I
        bounds.SOC_prev       = SOC
        bounds.T_prev         = T
        bounds.c_s_n_prev     = c_s_n_max
        bounds.c_e_min_prev   = c_e_min
        bounds.η_plating_prev = η_plating
    end

    return nothing
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

check_appropriate_method(method::Symbol) = @assert method ∈ (:I, :P, :V)

@inline function instant_hit_bounds(model::model_output, opts::options_model)
    var_keep = opts.var_keep
    
    fields = fieldnames(model_output)
    types = fieldtypes(model_output)
    @inbounds for (field,_type) in zip(fields,types)
        if _type <: AbstractArray{Float64} && getproperty(var_keep, field)
            x = getproperty(model, field)
            if length(x) > 1
                deleteat!(x, length(x))
            end
        end
    end
end

@inline function check_solve(run::Union{run_constant,run_residual}, model::R1, int::R2, p, bounds, opts::R5, funcs, keep_Y::Bool, iter::Int64, Y::Vector{Float64}, t::Float64) where {R1<:model_output,R2<:Sundials.IDAIntegrator,R5<:options_model}
    if t === int.tprev
        # Sometimes the initial step at t = 0 can be too large. This reduces the step size
        if t === 0.0
            if iter === 2
                Sundials.IDASetInitStep(int.mem,opts.reltol)
            else
                error("Model failed to converge at t = $(t). Try tightening the absolute and relative tolerances.")
            end
        else
            error("Model failed to converge at t = $(t)")
        end
    elseif iter === opts.maxiters
        error("Reached max iterations of $(opts.maxiters) at t = $(t)")
    elseif within_bounds(run)
        # update Y only after checking the stop conditions. this is done to store a copy of the
        # previous model run in case any back-interpolation is needed
        set_var!(model.Y, keep_Y, keep_Y ? copy(Y) : Y)
    else # no errors and run.info.flag ≠ -1
        return false
    end

    return true
end

@inline function check_solve(run::run_function, model::R1, int::R2, p::param, bounds::boundary_stop_conditions, opts::R5, funcs, keep_Y::Bool, iter::Int64, Y::Vector{Float64}, t::Float64) where {R1<:model_output,R2<:Sundials.IDAIntegrator,R5<:options_model}
    if iter === opts.maxiters
        error("Reached max iterations of $(opts.maxiters) at t = $(int.t)")
    elseif within_bounds(run)
        # update Y only after checking the stop conditions. this is done to store a copy of the
        # previous model run in case any back-interpolation is needed
        set_var!(model.Y, keep_Y, keep_Y ? copy(Y) : Y)
        
        # check to see if the run needs to be reinitialized
        if t - int.tprev < 1e-3opts.reltol
            check_reinitialization!(model, int, run, p, bounds, opts, funcs)
        end

        return true
    else # no errors and run.info.flag ≠ -1
        return false
    end
end

@inline function check_reinitialization!(model::R1, int::R2, run::R3, p::R4, bounds::R5, opts::R6, funcs) where {R1<:model_output, R2<:Sundials.IDAIntegrator, R3<:AbstractRun,R4<:param,R5<:boundary_stop_conditions,R6<:options_model}
    """
    Checking the current function for discontinuities.
    If there is a significant change in current after a step size of dt = reltol,
    then rerun Newton's method and reinitialize at this new time step.
    """
    
    Y = int.u.v
    YP = int.du.v
    t_new = int.t + opts.reltol

    value_old = value(run)
    value_new = run.func(t_new,Y,YP,p)
    
    if !≈(value_old, value_new, atol=opts.abstol, rtol=opts.reltol)
        initialize_states!(p,Y,YP,run,opts,funcs,(@inbounds model.SOC[end]); t=t_new)
        #run.value .= value_new

        Sundials.IDAReInit(int.mem, t_new, Y, YP)
    end
    return nothing
end

@inline function check_errors_parameters_runtime(p::R1,opts::R2) where {R1<:param,R2<:options_model}
    ϵ_sp, ϵ_sn = active_material(p)

    if ( ϵ_sp > 1 ) error("ϵ_p + ϵ_fp must be ∈ [0, 1)") end
    if ( ϵ_sn > 1 ) error("ϵ_n + ϵ_fn must be ∈ [0, 1)") end
    if ( p.θ[:θ_max_p] > p.θ[:θ_min_p] ) error("θ_max_p must be < θ_min_p") end
    if ( p.θ[:θ_min_n] > p.θ[:θ_max_n] ) error("θ_min_n must be < θ_max_n") end

    return nothing
end

function check_errors_initial(θ, numerics, N)
    if !(numerics.jacobian ∈ (:symbolic, :AD))
        error("`jacobian` can either be :symbolic or :AD")
    end

    return nothing
end

check_is_hold(x::Symbol,model::model_output) = (x===:hold) && (!isempty(model) ? true : error("Cannot use `:hold` without a previous model."))
check_is_hold(x,model) = false