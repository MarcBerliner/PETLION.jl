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
