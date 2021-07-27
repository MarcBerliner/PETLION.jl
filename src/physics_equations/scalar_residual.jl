@inline calc_V(Y::Vector{<:Number}, p::AbstractParam, ind_Φ_s::T=p.ind.Φ_s) where {T<:AbstractUnitRange{Int64}} = @inbounds Y[ind_Φ_s[1]] - Y[ind_Φ_s[end]]
@inline calc_I(Y::Vector{<:Number}, p::AbstractParam) = @inbounds Y[end]
@inline calc_P(Y::Vector{<:Number}, p::AbstractParam) = calc_I(Y,p)*calc_I1C(p)*calc_V(Y,p)

@inline method_I(Y, p) = calc_I(Y,p)
@inline method_V(Y, p) = calc_V(Y,p)
@inline method_P(Y, p) = calc_P(Y,p)

@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::param,run::run_constant{method}) where {method<:AbstractMethod,T1<:Float64,T2<:Vector{T1}} = res[end] = method(Y,p) - value(run)
@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::param,run::run_function{method,func}) where {method<:AbstractMethod,T1<:Float64,T2<:Vector{T1},func<:Function} = @inbounds (res[end] = method(Y,p) - (run.func)(t,Y,YP,p)::Float64)
@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::param,run::run_residual{func}) where {T1<:Float64,T2<:Vector{T1},func<:Function} = @inbounds ((run.value .= Y[end]); res[end] = (run.func)(t,Y,YP,p))

@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::AbstractParam,run::run_constant{method}) where {method<:AbstractMethod,T1<:Num,T2<:Vector{T1}} = res[end] = method(Y,p) - value(run)
@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::AbstractParam,run::run_function{method,func}) where {method<:AbstractMethod,T1<:Num,T2<:Vector{T1},func<:Function} = @inbounds (res[end] = method(Y,p) - (run.func)(t,Y,YP,p))
@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::AbstractParam,run::run_residual{func}) where {T1<:Num,T2<:Vector{T1},func<:Function} = @inbounds (res[end] = (run.func)(t,Y,YP,p))

function get_jacobian_and_residuals(p::param, run::Union{run_function,run_residual,run_constant})
    θ_sym, Y, YP, t, SOC, I, γ, p_sym, θ_keys, θ_len = get_symbolic_vars(p)
    res = similar(Y)
    scalar_residual!(res,t,Y,YP,p_sym,run)

    J_Y = @inbounds ModelingToolkit.sparsejacobian([res[end]], Y)[:]
    J_YP = @inbounds γ.*ModelingToolkit.sparsejacobian([res[end]], YP)[:]

    scalar_contains_differential = !isempty(J_YP.nzval)

    J_vec = J_Y .+ J_YP
    J_sp_scalar = convert(SparseVector{Float64, Int64}, sparse(.!iszero.(J_vec)))
    scalar_func = eval(build_function(J_vec.nzval,t,Y,YP,γ,θ_sym)[2])

    N = p.N.tot
    J_sp_base = p.funcs[:I].J_y!.sp
    base_func = p.funcs[:I].J_y!.func
    J_sp = [J_sp_base; J_sp_scalar']

    ind_base   = findall(J_sp.rowval .< N)
    ind_scalar = findall(J_sp.rowval .== N)
    if ind_base == 1:length(ind_base)
        ind_base = 1:length(ind_base)
    end

    J_base   = @views @inbounds J_sp.nzval[ind_base]
    J_scalar = @views @inbounds J_sp.nzval[ind_scalar]

    J = jacobian_combined(J_sp,base_func,J_base,scalar_func,J_scalar,scalar_contains_differential)

    """
    Residuals section
    """
    R_scalar_func = @inbounds eval(build_function(res[end],t,Y,YP,θ_sym))

    R_base_func = p.funcs[:I].f!

    R = residuals_combined(R_base_func,R_scalar_func)

    return J,R
end