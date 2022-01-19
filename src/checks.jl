@inline function check_simulation_stop!(sol, t::Float64, Y, YP, run::AbstractRun, p, bounds, opts::options_simulation_immutable{T};
    ϵ::Float64 = t < 1.0 ? opts.reltol : 0.0,
    ) where T<:Function
    
    if t ≥ run.tf
        run.info.flag = 0
        run.info.exit_reason = "Final time reached"
        return nothing
    end

    # continue checking the bounds or return after only evaluating the time
    !opts.check_bounds && (return nothing)

    I = calc_I(Y,p)
    
    check_stop_I(         p, run, sol, Y, YP, bounds, ϵ, I)
    check_stop_V(         p, run, sol, Y, YP, bounds, ϵ, I)
    check_stop_SOC(       p, run, sol, Y, YP, bounds, ϵ, I)
    check_stop_T(         p, run, sol, Y, YP, bounds, ϵ, I)
    check_stop_c_s_surf(  p, run, sol, Y, YP, bounds, ϵ, I)
    check_stop_c_e(       p, run, sol, Y, YP, bounds, ϵ, I)
    check_stop_η_plating( p, run, sol, Y, YP, bounds, ϵ, I)
    check_stop_dfilm(     p, run, sol, Y, YP, bounds, ϵ, I)
    opts.stop_function(   p, run, sol, Y, YP, bounds, ϵ, I)

    return nothing
end

@inline check_stop_I(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, R3<:AbstractRun{method_I},R4<:model,R5<:boundary_stop_conditions_immutable} = nothing
@inline function check_stop_I(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, method, R3<:AbstractRun{method},R4<:model,R5<:boundary_stop_conditions_immutable}
    
    if (I - bounds.I_max > ϵ)
        t_frac = (bounds.prev.I - bounds.I_max)/(bounds.prev.I - I)
        if t_frac < bounds.prev.t_final_interp_frac
            bounds.prev.t_final_interp_frac = t_frac
            run.info.flag = 7
            run.info.exit_reason = "Above maximum permitted C-rate"
        end
    elseif (bounds.I_min - I > ϵ)
        t_frac = (bounds.prev.I - bounds.I_min)/(bounds.prev.I - I)
        if t_frac < bounds.prev.t_final_interp_frac
            bounds.prev.t_final_interp_frac = t_frac
            run.info.flag = 8
            run.info.exit_reason = "Below minimum permitted C-rate"
        end
    end
    bounds.prev.I = I    

    return nothing
end

@inline check_stop_V(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
) where {R2<:Vector{Float64}, R3<:AbstractRun{method_V},R4<:model,R5<:boundary_stop_conditions_immutable} = nothing
@inline function check_stop_V(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, method, R3<:AbstractRun{method},R4<:model,R5<:boundary_stop_conditions_immutable}
    
    V = calc_V(Y,p)
    if (bounds.V_min - V > ϵ) && I < 0
        t_frac = (bounds.prev.V - bounds.V_min)/(bounds.prev.V - V)
        if t_frac < bounds.prev.t_final_interp_frac
            bounds.prev.t_final_interp_frac = t_frac
            run.info.flag = 1
            run.info.exit_reason = "Below minimum voltage limit"
        end
    elseif (V - bounds.V_max > ϵ) && I > 0
        t_frac = (bounds.prev.V - bounds.V_max)/(bounds.prev.V - V)
        if t_frac < bounds.prev.t_final_interp_frac
            bounds.prev.t_final_interp_frac = t_frac
            run.info.flag = 2
            run.info.exit_reason = "Above maximum voltage limit"
        end
    end
    bounds.prev.V = V

    return nothing
end

@inline function check_stop_SOC(p::R4, run::R3, sol::solution, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, R3<:AbstractRun,R4<:model,R5<:boundary_stop_conditions_immutable}
    
    SOC = @inbounds sol.SOC[end]
    if (bounds.SOC_min - SOC > ϵ) && I < 0
        t_frac = (bounds.prev.SOC - bounds.SOC_min)/(bounds.prev.SOC - SOC)
        if t_frac < bounds.prev.t_final_interp_frac
            bounds.prev.t_final_interp_frac = t_frac
            run.info.flag = 3
            run.info.exit_reason = "Below minimum SOC limit"
        end
    elseif (SOC - bounds.SOC_max > ϵ) && I > 0
        t_frac = (bounds.prev.SOC - bounds.SOC_max)/(bounds.prev.SOC - SOC)
        if t_frac < bounds.prev.t_final_interp_frac
            bounds.prev.t_final_interp_frac = t_frac
            run.info.flag = 4
            run.info.exit_reason = "Above maximum SOC limit"
        end
    end
    bounds.prev.SOC = SOC

    return SOC
end

@inline check_stop_T(p::model_temp{false}, run, sol, Y, YP, bounds, ϵ, I) = nothing
@inline function check_stop_T(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, R3<:AbstractRun,R4<:model_temp{true},R5<:boundary_stop_conditions_immutable}

    if !isnan(bounds.T_max)
        T = temperature_weighting(calc_T(Y,p),p)
        if T - bounds.T_max > ϵ
            t_frac = (bounds.prev.T - bounds.T_max)/(bounds.prev.T - T)
            if t_frac < bounds.prev.t_final_interp_frac
                bounds.prev.t_final_interp_frac = t_frac
                run.info.flag = 5
                run.info.exit_reason = "Above maximum permitted temperature"
            end
        end
        bounds.prev.T = T
    end

    return nothing
end

@inline function c_s_n_maximum(Y::Vector{Float64},p::model_solid_diff{:Fickian})
    c_s_n_max = -Inf
    @inbounds for i in 1:p.N.n
        @inbounds c_s_n_max = max(c_s_n_max, Y[p.ind.c_s_avg.n[p.N.r_n*i]])
    end
    return c_s_n_max
end
@inline function c_s_n_maximum(Y::Vector{Float64},p::Union{model_solid_diff{:polynomial},model_solid_diff{:quadratic}})
    c_s_n_max = -Inf
    @inbounds for ind in p.ind.c_s_avg.n
        @inbounds c_s_n_max = max(c_s_n_max, Y[ind])
    end
    return c_s_n_max
end

@inline function check_stop_c_s_surf(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, R3<:AbstractRun,R4<:model,R5<:boundary_stop_conditions_immutable}
    
    if !isnan(bounds.c_s_n_max)
        c_s_n_max = c_s_n_maximum(Y,p)

        if I > 0
            if c_s_n_max - bounds.c_s_n_max*p.θ[:c_max_n] > ϵ
                t_frac = (bounds.prev.c_s_n - bounds.c_s_n_max*(p.θ[:c_max_n]))/(bounds.prev.c_s_n - c_s_n_max)
                if t_frac < bounds.prev.t_final_interp_frac
                    bounds.prev.t_final_interp_frac = t_frac
                    run.info.flag = 6
                    run.info.exit_reason = "Above c_s_n saturation threshold"
                end
            end
        end
        bounds.prev.c_s_n = c_s_n_max
    end

    return nothing
end

@inline function check_stop_c_e(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, R3<:AbstractRun, R4<:model, R5<:boundary_stop_conditions_immutable}
    
    if !isnan(bounds.c_e_min)
        c_e_min = +Inf
        @inbounds for ind in p.ind.c_e
            @inbounds c_e_min = min(c_e_min, Y[ind])
        end
        if bounds.c_e_min - c_e_min > ϵ
            t_frac = (bounds.prev.c_e_min - bounds.c_e_min)/(bounds.prev.c_e_min - c_e_min)
            if t_frac < bounds.prev.t_final_interp_frac
                bounds.prev.t_final_interp_frac = t_frac
                run.info.flag = 9
                run.info.exit_reason = "Below minimum permitted c_e"
            end
        end
        bounds.prev.c_e_min = c_e_min
    end

    return nothing
end

@inline function check_stop_η_plating(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, R3<:AbstractRun, R4<:model, R5<:boundary_stop_conditions_immutable}
    
    η_plating = calc_η_plating(Y,p)
    if !isnan(bounds.η_plating_min) && bounds.η_plating_min - η_plating > ϵ
        t_frac = (bounds.prev.η_plating - bounds.η_plating_min)/(bounds.prev.η_plating - η_plating)
        if t_frac < bounds.prev.t_final_interp_frac
            bounds.prev.t_final_interp_frac = t_frac
            run.info.flag = 9
            run.info.exit_reason = "Below minimum permitted η_plating"
        end
    end
    bounds.prev.η_plating = η_plating

    return nothing
end

@inline check_stop_dfilm(::model_age{false}, run, sol, Y, YP, bounds, ϵ, I) = nothing
@inline function check_stop_dfilm(p::R4, run::R3, sol, Y::R2, YP::R2, bounds::R5, ϵ::Float64, I::Float64
    ) where {R2<:Vector{Float64}, R3<:AbstractRun, R4<:model_age{:SEI}, R5<:boundary_stop_conditions_immutable}
    
    dfilm_max = -Inf
    @inbounds for i in 1:p.N.n
        @inbounds dfilm_max = max(dfilm_max, YP[p.ind.film[i]])
    end
    
    if !isnan(bounds.dfilm_max) && dfilm_max - bounds.dfilm_max > ϵ
        t_frac = (bounds.prev.dfilm - bounds.dfilm_max)/(bounds.prev.dfilm - dfilm_max)
        if t_frac < bounds.prev.t_final_interp_frac
            bounds.prev.t_final_interp_frac = t_frac
            run.info.flag = 10
            run.info.exit_reason = "Above maximum film growth rate"
        end
    end
    bounds.prev.dfilm = dfilm_max

    return nothing
end



@inline function check_solve(run::run_constant, sol::R1, int::R2, p, bounds, opts::R5, funcs, keep_Y::Bool, iter::Int64, Y::Vector{Float64}, t::Float64) where {R1<:solution,R2<:Sundials.IDAIntegrator,R5<:AbstractOptionsModel}
    if t == int.tprev
        # Sometimes the initial step at t = 0 can be too large. This reduces the step size
        if t == 0.0
            if iter == 2
                Sundials.IDASetInitStep(int.mem,opts.reltol)
            else
                error("Model failed to converge at t = $(t). Try tightening the absolute and relative tolerances.")
            end
        else
            error("Model failed to converge at t = $(t)")
        end
    elseif iter == opts.maxiters
        error("Reached max iterations of $(opts.maxiters) at t = $(t)")
    elseif within_bounds(run)
        # update Y only after checking the stop conditions. this is done to store a copy of the
        # previous sol run in case any back-interpolation is needed
        set_var!(sol.Y, keep_Y ? copy(Y) : Y, keep_Y)
    else # no errors and run.info.flag ≠ -1
        return false
    end

    return true
end

@inline function check_solve(run::run_function, sol::R1, int::R2, p::model, bounds::boundary_stop_conditions_immutable, opts::R5, funcs, keep_Y::Bool, iter::Int64, Y::Vector{Float64}, t::Float64) where {R1<:solution,R2<:Sundials.IDAIntegrator,R5<:AbstractOptionsModel}
    if iter == opts.maxiters
        error("Reached max iterations of $(opts.maxiters) at t = $(int.t)")
    elseif within_bounds(run)
        # update Y only after checking the stop conditions. this is done to store a copy of the
        # previous sol run in case any back-interpolation is needed
        set_var!(sol.Y, keep_Y ? copy(Y) : Y, keep_Y)
        
        # check to see if the run needs to be reinitialized
        if t - int.tprev < 1e-3opts.reltol
            check_reinitialization!(sol, int, run, p, bounds, opts, funcs)
        end

        return true
    else # no errors and run.info.flag ≠ -1
        return false
    end
end

@inline function check_input_arguments(names::Tuple{Symbol})
    """
    Checking if the inputs to the simulation are valid.
    Try/catch block in `assess_input` finds any method errors
    """
    name = @inbounds names[1]
    return name
end
@inline function check_input_arguments(names::T) where T<:Tuple
    """
    There are more 0 or >1 input arguments, which is invalid
    """

    # Tuple of all methods defined by AbstractMethod
    valid_methods = (method_symbol.(subtypes(AbstractMethod))...,)

    input_methods = Symbol[]
    invalid_args = Symbol[]
    @inbounds for name in names
        if name ∈ valid_methods
            push!(input_methods, name)
        else
            push!(invalid_args, name)
        end
    end
    input_methods = (input_methods...,)
    invalid_args = (invalid_args...,)
        
    if length(input_methods) === 0
        str_methods = replace("$(valid_methods)", ":"=>"")
        error("ERROR\n--------\n" *
        "  No inputs are selected, choose one from: $str_methods")
    end

    # If there are more than one names, then that means that
    # (a) ≥1 inputs are selected, or (b) an invalid input is selected
    str = "ERROR\n--------\n"

    if length(input_methods) > 1
        str_methods = replace("$(valid_methods)", ":"=>"")
        str *= "  Cannot select more than one input from: $(str_methods)"
    end
        
    if length(invalid_args) ≥ 1 && length(input_methods) > 1
        str *= "\n"
    end

    if length(invalid_args) ≥ 1
        str *= "  Invalid keyword argument"
        str *= length(invalid_args) == 1 ? ": " : "s: "
        str_args = replace(length(invalid_args) == 1 ? "$(invalid_args[1])" : "$(invalid_args)", ":"=>"")
        str *= str_args
    end

    error(str)
end

@inline function check_reinitialization!(sol::R1, int::R2, run::R3, p::R4, bounds::R5, opts::R6, funcs) where {R1<:solution, R2<:Sundials.IDAIntegrator, R3<:AbstractRun,R4<:model,R5<:boundary_stop_conditions_immutable,R6<:AbstractOptionsModel}
    """
    Checking the current function for discontinuities.
    If there is a significant change in current after a step size of dt = reltol,
    then rerun Newton's method and reinitialize at this new time step.
    """
    
    Y = int.u.v
    YP = int.du.v
    # take a step of Δt = the relative tolerance
    t_new = int.t + opts.reltol

    value_old = value(run)
    value_new = run.func(t_new,Y,YP,p)
    
    # if the function values at t vs. t + Δt are very different (i.e., there is a discontinuity)
    # then reinitialize the DAE at t + Δt
    if !≈(value_old, value_new, atol=opts.abstol, rtol=opts.reltol)
        initialize_states!(p,Y,YP,run,opts,funcs,(@inbounds sol.SOC[end]); t=t_new)

        Sundials.IDAReInit(int.mem, t_new, Y, YP)
    end
    return nothing
end

@inline function check_errors_parameters_runtime(p::R1,opts::R2) where {R1<:model,R2<:AbstractOptionsModel}
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

check_is_hold(x::Symbol,sol::solution) = (x == :hold) && (!isempty(sol) ? true : error("Cannot use `:hold` without a previous sol."))
check_is_hold(::Any,::solution) = false