"""
Current
"""

method_symbol(::Type{method_I}) = :I

@inline input_method(::Val{:I}, input, y...) = (method_I(), input)

@inline method_I(Y, p)   = calc_I(Y,p)

@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},sol, res_I_guess) where {method<:method_I,in<:Number}
    input = run.input
    @inbounds run.value[] = Y0[p.ind.I[1]] = input
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},sol::solution, res_I_guess) where {method<:method_I,in<:Symbol}
    input = run.input
    if check_is_hold(input,sol)
        @inbounds run.value[] = Y0[p.ind.I[1]] = calc_I((@views @inbounds sol.Y[end]), p)
    elseif input == :rest
        @inbounds run.value[] = Y0[p.ind.I[1]] = 0.0
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0::Vector{Float64},p,run::run_function{method,func},sol, res_I_guess) where {method<:method_I,func<:Function}
    run.value[] = Y0[p.ind.I[1]] = run.func(0.0,Y0,YP0,p)
    return nothing
end

"""
Voltage
"""

method_symbol(::Type{method_V}) = :V

@inline input_method(::Val{:P}, input, y...) = (method_P(), input)

@inline method_V(Y, p)   = calc_V(Y,p)

@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},sol, res_I_guess) where {method<:method_P,in<:Number}
    @inbounds run.value[] = input = run.input

    @inbounds Y0[p.ind.I[1]] = input/(calc_V(Y0,p)*p.θ[:I1C])
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},sol::solution, res_I_guess) where {method<:method_P,in<:Symbol}
    @inbounds run.value[] = input = run.input
    if check_is_hold(input,sol)

        @inbounds Y0[p.ind.I[1]] = calc_P((@views @inbounds sol.Y[end]), p)
    elseif input == :rest
        @inbounds run.value[] = Y0[p.ind.I[1]] = 0.0
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0::Vector{Float64},p,run::run_function{method,func},sol, res_I_guess) where {method<:method_P,func<:Function}
    run.value[] = Y0[p.ind.I[1]] = run.func(0.0,Y0,YP0,p)/(calc_V(Y0,p)*p.θ[:I1C])
    return nothing
end

"""
Power
"""

method_symbol(::Type{method_P}) = :P

@inline input_method(::Val{:V}, input, y...) = (method_V(), input)

@inline method_P(Y, p)   = calc_P(Y,p)

@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},sol::solution, res_I_guess) where {method<:method_V,in<:Number}
    input = run.input
    @inbounds run.value[] = input
    if !isempty(sol) && (I_prev = calc_I((@views @inbounds sol.Y[end]), p); I_prev ≠ 0)
        @inbounds Y0[p.ind.I[1]] = I_prev
    else
        OCV = calc_V(Y0,p)
        @inbounds Y0[p.ind.I[1]] = input > OCV ? +1.0 : -1.0
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},sol::solution, res_I_guess) where {method<:method_V,in<:Symbol}
    input = run.input
    if check_is_hold(input,sol)
        Y = @views @inbounds sol.Y[end]
        @inbounds run.value[] = calc_V(Y, p)
        @inbounds Y0[p.ind.I[1]] = calc_V(Y, p)
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_function{method,func},sol::solution, res_I_guess) where {method<:method_V,func<:Function}
    @inbounds run.value[] = run.func(0.0,Y0,YP0,p)
    if !isempty(sol)
        @inbounds Y0[p.ind.I[1]] = calc_I((@views @inbounds sol.Y[end]), p)
    else
        # Arbitrary guess for the initial current. 
        OCV = calc_V(Y0,p)
        @inbounds Y0[p.ind.I[1]] = value(run) > OCV ? +1.0 : -1.0
    end
    return nothing
end

"""
Lithium plating overpotential
"""

method_symbol(::Type{method_η_p}) = :η_p

@inline input_method(::Val{:η_p}, input, y...) = (method_η_p(), input)

@inline method_η_p(Y, p) = calc_η_plating(Y,p)


@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},sol::solution, res_I_guess) where {method<:method_η_p,in<:Number}
    input = run.input
    @inbounds run.value[] = input
    if !isempty(sol)
        @inbounds Y0[p.ind.I[1]] = calc_I((@views @inbounds sol.Y[end]), p)
    else
        OCV = calc_V(Y0,p)
        @inbounds Y0[p.ind.I[1]] = input > OCV ? +1.0 : -1.0
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_constant{method,in},sol::solution, res_I_guess) where {method<:method_η_p,in<:Symbol}
    input = run.input
    if check_is_hold(input,sol)
        Y = @views @inbounds sol.Y[end]
        val = calc_η_plating(Y, p)
        @inbounds run.value[] = val
        @inbounds Y0[p.ind.I[1]] = val
    else
        error("Unsupported input symbol.")
    end
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_function{method,func},sol::solution, res_I_guess) where {method<:method_η_p,func<:Function}
    @inbounds run.value[] = run.func(0.0,Y0,YP0,p)
    if !isempty(sol)
        @inbounds Y0[p.ind.I[1]] = calc_I((@views @inbounds sol.Y[end]), p)
    else
        # Arbitrary guess for the initial current. 
        OCV = calc_V(Y0,p)
        @inbounds Y0[p.ind.I[1]] = value(run) > OCV ? +1.0 : -1.0
    end
    return nothing
end

"""
User-defined residual
"""

method_symbol(::Type{method_res}) = :res

@inline method_res(Y, p) = 0.0

@inline function input_method(::Val{:res}, input::T, p::model, sol::solution) where T
    res = custom_res!(p,input,sol)
    
    return method_res(), res
end

@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_residual{method,func},sol::solution, res_I_guess::Number) where {method<:method_res,func<:Function}
    @inbounds Y0[end] = res_I_guess
    return nothing
end
@inline function initial_current!(Y0::Vector{Float64},YP0,p,run::run_residual{method,func},sol::solution, res_I_guess::Nothing) where {method<:method_res,func<:Function}
    @inbounds Y0[end] = isempty(sol.Y) ? 1.0 : sol.Y[end][p.ind.I[1]]
    return nothing
end

"""
Temperature rate of change
"""

@inline function input_method(::Val{:dT}, input::T, p::model, sol::solution) where T
    if !(p.numerics.temperature == true)
        error("Temperature must be enabled when using `dT`.")
    end
    func = custom_res!(p,input,constant_temperature,sol)
    
    return method_res(), func
end

"""
Concentration rate of change
"""

@inline function input_method(::Val{:dc_s_p_max}, input::T, p::model, sol::solution) where T
    @assert !isempty(sol.Y)
    ind_full = c_s_indices(p,:p;surf=true)
    ind = ind_full[argmax((@views @inbounds sol.Y[end][ind_full]))]
    
    func = custom_res!(p,input,state_deriv_func(ind),sol)
    return method_res(), func
end

@inline function input_method(::Val{:dc_s_p_min}, input::T, p::model, sol::solution) where T
    @assert !isempty(sol.Y)
    ind_full = c_s_indices(p,:p;surf=true)
    ind = ind_full[argmin((@views @inbounds sol.Y[end][ind_full]))]
    
    func = custom_res!(p,input,state_deriv_func(ind),sol)
    return method_res(), func
end

@inline function input_method(::Val{:dc_s_n_max}, input::T, p::model, sol::solution) where T
    @assert !isempty(sol.Y)
    ind_full = c_s_indices(p,:n;surf=true)
    ind = ind_full[argmax((@views @inbounds sol.Y[end][ind_full]))]
    
    func = custom_res!(p,input,state_deriv_func(ind),sol)
    return method_res(), func
end

@inline function input_method(::Val{:dc_s_n_min}, input::T, p::model, sol::solution) where T
    @assert !isempty(sol.Y)
    ind_full = c_s_indices(p,:n;surf=true)
    ind = ind_full[argmin((@views @inbounds sol.Y[end][ind_full]))]
    
    func = custom_res!(p,input,state_deriv_func(ind),sol)
    return method_res(), func
end

@inline function input_method(::Val{:dc_e_max}, input::T, p::model, sol::solution) where T
    @assert !isempty(sol.Y)
    ind_full = p.ind.c_e
    ind = ind_full[argmax((@views @inbounds sol.Y[end][ind_full]))]
    
    func = custom_res!(p,input,state_deriv_func(ind),sol)
    return method_res(), func
end

@inline function input_method(::Val{:dc_e_min}, input::T, p::model, sol::solution) where T
    @assert !isempty(sol.Y)
    ind_full = p.ind.c_e
    ind = ind_full[argmin((@views @inbounds sol.Y[end][ind_full]))]
    
    func = custom_res!(p,input,state_deriv_func(ind),sol)
    return method_res(), func
end


# For the differential terms, create another input without the `d` in front which
# only accept an input of `:hold` and sets dx = 0.
for x in (:T,)#:c_s_p_max,:c_s_p_min,:c_s_n_max,:c_s_n_min,:c_e_max,:c_e_min)
    str = "@inline input_method(::Val{:$x}, input::Symbol, x...) = input ≠ :hold ? error(\"This is a differential state: the only valid input for `$(x)` is `:hold`. Use `d$(x)` for more complex protocols.\") : input_method(Val(:d$(x)), input, x...)"
    
    str = remove_module_name(str)
    eval(Meta.parse(str))
end