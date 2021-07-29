@inline calc_V(Y::Vector{<:Number}, p::AbstractParam) = @inbounds Y[p.ind.Φ_s[1]] - Y[p.ind.Φ_s[end]]
@inline calc_I(Y::Vector{<:Number}, p::AbstractParam) = @inbounds Y[end]
@inline calc_P(Y::Vector{<:Number}, p::AbstractParam) = calc_I(Y,p)*p.θ[:I1C]*calc_V(Y,p)

@inline method_I(Y, p) = calc_I(Y,p)
@inline method_V(Y, p) = calc_V(Y,p)
@inline method_P(Y, p) = calc_P(Y,p)

@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::AbstractParam,run::run_constant{method}) where {method<:AbstractMethod,T1<:Number,T2<:Vector{T1}} = @inbounds (res[end] = method(Y,p) - value(run))
@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::AbstractParam,run::run_function{method,func}) where {method<:AbstractMethod,T1<:Number,T2<:Vector{T1},func<:Function} = @inbounds (res[end] = method(Y,p) - run.func(t,Y,YP,p))
@inline scalar_residual!(res::T2,t::T1,Y::T2,YP::T2,p::AbstractParam,run::run_residual{func}) where {T1<:Number,T2<:Vector{T1},func<:Function} = @inbounds (res[end] = run.func(t,Y,YP,p))

SM_nzval = SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}
@inline @inbounds function scalar_jacobian!(J::SM_nzval,::T1,Y::T2,::T2,γ::T1,p::param,::run_constant{method_I}) where {T1<:Float64,T2<:Vector{T1}}
    J[1] = 1.0
    return nothing
end
@inline @inbounds function scalar_jacobian!(J::SM_nzval,::T1,Y::T2,::T2,γ::T1,p::param,::run_constant{method_V}) where {T1<:Float64,T2<:Vector{T1}}
    J[1] = 1.0
    J[2] = -1.0
    return nothing
end
@inline @inbounds function scalar_jacobian!(J::SM_nzval,::T1,Y::T2,::T2,γ::T1,p::param,::run_constant{method_P}) where {T1<:Float64,T2<:Vector{T1}}
    I1C = p.θ[:I1C]
    I = calc_I(Y,p)*I1C
    V = calc_V(Y,p)
    J[1] = I
    J[2] = -I
    J[3] = V*I1C
    return nothing
end

@inbounds function get_jacobian_sparsity(p::param, ::run_constant{method}) where method<:method_I
    J = spzeros(Float64,p.N.tot)
    J[p.ind.I] .= 1

    return J
end
@inbounds function get_jacobian_sparsity(p::param, ::run_constant{method}) where method<:method_V
    J = spzeros(Float64,p.N.tot)
    J[p.ind.Φ_s[[1,end]]] .= 1

    return J
end
@inbounds function get_jacobian_sparsity(p::param, ::run_constant{method}) where method<:method_P
    J = spzeros(Float64,p.N.tot)
    J[p.ind.I] .= 1
    J[p.ind.Φ_s[[1,end]]] .= 1

    return J
end

function _get_method_funcs(p::param, run::T) where T<:Union{run_function,run_residual}
    θ_sym, Y, YP, t, SOC, I, γ, p_sym, θ_keys = get_symbolic_vars(p)
    res = similar(Y)
    scalar_residual!(res,t,Y,YP,p_sym,run)

    J_Y  = @inbounds ModelingToolkit.sparsejacobian([res[end]], Y)[:]
    J_YP = @inbounds γ.*ModelingToolkit.sparsejacobian([res[end]], YP)[:]
    scalar_contains_differential = !isempty(J_YP.nzval)

    residual_is_differentiable = true

    J_vec = J_Y .+ J_YP
    J_sp_scalar = convert(SparseVector{Float64, Int64}, sparse(.!iszero.(J_vec)))

    """
    Updating the theta vector to ensure any new parameters will be accounted for
    """
    θ_keys_scalar = get_only_θ_used_in_model(θ_sym, θ_keys, res[end])[2]

    θ_keys = deepcopy(p.cache.θ_keys)
    @inbounds for key in θ_keys_scalar
        if !(key ∈ θ_keys)
            push!(θ_keys, key)
        end
    end

    θ_tot      = [p.θ[key] for key in θ_keys]
    θ_sym_slim = [p_sym.θ[key] for key in θ_keys]
    update_θ!(θ_tot,θ_keys,p.θ)

    J_scalar_func = eval(build_function(J_vec.nzval,t,Y,YP,γ,θ_sym_slim)[2])

    J_sp_base = p.funcs.J_y!.sp
    J_base_func = p.funcs.J_y!.func

    """
    Creating the scalar function
    """
    res_algebraic = @inbounds res[end]
    
    if scalar_contains_differential
        vars_in_residual = @inbounds ModelingToolkit.get_variables(res[end])
        # Find all indices where YP is used in the residual equation
        ind_differential = findall(in(vars_in_residual), @views @inbounds YP[1:p.N.diff])
        
        @assert !isempty(ind_differential)

        residuals_PET!(res,t,Y,YP,p_sym)
        @inbounds for ind in ind_differential
            res_algebraic = substitute(res_algebraic, Dict(YP[ind] => res[ind] + YP[ind]))
        end

        scalar_residual_alg_func = eval(build_function(res_algebraic,t,Y,YP,θ_sym_slim))
        scalar_residal_alg! = scalar_residual_alg_func
        J_alg_vec = ModelingToolkit.sparsejacobian([res_algebraic], Y[p.N.diff+1:end])[:]
        
        J_scalar_alg_func = eval(build_function(J_alg_vec.nzval,t,Y,YP,γ,θ_sym_slim)[2])
        J_sp_alg_scalar = convert(SparseVector{Float64, Int64}, sparse(.!iszero.(J_alg_vec)))
    else
        scalar_residal_alg! = scalar_residual!

        J_scalar_alg_func = J_scalar_func
        J_sp_alg_scalar = J_sp_scalar[p.N.diff+1:end]
    end

    return combine_Jac_and_res(p,J_sp_base,J_base_func,J_sp_scalar,J_scalar_func,θ_tot,θ_keys,scalar_residal_alg!,J_scalar_alg_func,J_sp_alg_scalar,residual_is_differentiable)
end

function _get_method_funcs(p::param, run::run_constant{method}) where method<:AbstractMethod
    J_sp_scalar = get_jacobian_sparsity(p,run)

    θ_tot = p.cache.θ_tot
    θ_keys = p.cache.θ_keys
    update_θ!(θ_tot,θ_keys,p.θ)

    residual_is_differentiable = true
    
    J_scalar_func = scalar_jacobian!
    J_sp_base = p.funcs.J_y!.sp
    J_base_func = p.funcs.J_y!.func

    scalar_residal_alg! = scalar_residual!
    J_scalar_alg_func = J_scalar_func
    J_sp_alg_scalar = J_sp_scalar[p.N.diff+1:end]
    
    return combine_Jac_and_res(p,J_sp_base,J_base_func,J_sp_scalar,J_scalar_func,θ_tot,θ_keys,scalar_residal_alg!,J_scalar_alg_func,J_sp_alg_scalar,residual_is_differentiable)
end

function combine_Jac_and_res(p,J_sp_base,J_base_func,J_sp_scalar,J_scalar_func,θ_tot,θ_keys,scalar_residal_alg!,J_scalar_alg_func,J_sp_alg_scalar,residual_is_differentiable)
    J_full = _get_jacobian_combined(J_sp_base,J_base_func,J_sp_scalar,J_scalar_func,θ_tot,θ_keys)
    R_full = residual_combined(
        p.funcs.initial_conditions.f_diff!,
        p.funcs.initial_conditions.f_alg!,
        scalar_residual!,
        1:p.N.diff,
        p.N.diff+1:p.N.tot,
        θ_tot,
        θ_keys,
    )

    J_alg = _get_jacobian_combined((@inbounds J_sp_base[p.N.diff+1:end,p.N.diff+1:end]),p.funcs.initial_conditions.J_y_alg!.func,J_sp_alg_scalar,J_scalar_alg_func,θ_tot,θ_keys)
    R_diff = residual_combined(
        p.funcs.initial_conditions.f_diff!,
        emptyfunc,
        emptyfunc,
        1:p.N.diff,
        1:0,
        θ_tot,
        θ_keys,
    )
    R_alg = residual_combined(
        emptyfunc,
        p.funcs.initial_conditions.f_alg!,
        scalar_residal_alg!,
        1:0,
        1:p.N.alg,
        θ_tot,
        θ_keys,
    )

    return Jac_and_res(J_full,R_full,J_alg,R_diff,R_alg,Sundials.IDAIntegrator[])
end

function _get_jacobian_combined(J_sp_base,J_base_func,J_sp_scalar,J_scalar_func,θ_tot,θ_keys)
    J_sp = [J_sp_base; J_sp_scalar']

    ind_base   = findall(J_sp.rowval .< size(J_sp,1))
    ind_scalar = findall(J_sp.rowval .== size(J_sp,1))
    if ind_base == 1:length(ind_base)
        ind_base = 1:length(ind_base)
    end

    J_base   = @views @inbounds J_sp.nzval[ind_base]
    J_scalar = @views @inbounds J_sp.nzval[ind_scalar]

    J = jacobian_combined(J_sp,J_base_func,J_base,J_scalar_func,J_scalar,θ_tot,θ_keys)

    return J
end

function get_method_funcs!(p::param,run::run_constant{method}) where method<:AbstractMethod
    p.method_functions.Dict_constant[method] = _get_method_funcs(p,run)
end
function get_method_funcs!(p::param,run::run_function{method,func}) where {method<:AbstractMethod,func<:Function}
    if !haskey(p.method_functions.Dict_function,method)
        p.method_functions.Dict_function[method] = Dict{DataType,jacobian_combined}()
    end
    p.method_functions.Dict_function[method][func] = _get_method_funcs(p,run)
end
function get_method_funcs!(p::param,run::run_residual{func}) where {func<:Function}
    p.method_functions.Dict_residual[func] = _get_method_funcs(p,run)
end

"""
Multiple dispatch for the residuals function
"""
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:Function,T2<:Function,T3<:Function,T<:AbstractVector{Float64}}
    r.f_diff!((@views @inbounds res[r.ind_diff]), t, Y, YP, r.θ_tot)
    r.f_alg!( (@views @inbounds res[r.ind_alg]),  t, Y, YP, r.θ_tot)
    @inbounds res[end] = r.f_scalar!(t,Y,YP,r.θ_tot)
end
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:typeof(emptyfunc),T2<:Function,T3<:Function,T<:AbstractVector{Float64}}
    r.f_alg!(res,t,Y,YP,r.θ_tot)
    @inbounds res[end] = r.f_scalar!(t,Y,YP,r.θ_tot)
end
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:Function,T2<:Function,T3<:typeof(scalar_residual!),T<:AbstractVector{Float64}}
    r.f_diff!((@views @inbounds res[r.ind_diff]), t, Y, YP, r.θ_tot)
    r.f_alg!( (@views @inbounds res[r.ind_alg]),  t, Y, YP, r.θ_tot)
    scalar_residual!(res,t,Y,YP,p,run)
end
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:typeof(emptyfunc),T2<:Function,T3<:typeof(scalar_residual!),T<:AbstractVector{Float64}}
    r.f_alg!(res,t,Y,YP,r.θ_tot)
    scalar_residual!(res,t,Y,YP,p,run)
end
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:Function,T2<:typeof(emptyfunc),T3<:typeof(emptyfunc),T<:AbstractVector{Float64}}
    r.f_diff!(res, t, Y, YP, r.θ_tot)
end

"""
Multiple dispatch for the Jacobian function
"""
@inline function (J::jacobian_combined{T1,T2,T3})(t::Float64,Y::T,YP::T,γ::Float64,p,run) where {T<:Vector{Float64},T1<:Function,T2,T3<:Function}
    J.base_func(J.J_base,t,Y,YP,γ,J.θ_tot)
    J.scalar_func(J.J_scalar,t,Y,YP,γ,J.θ_tot)
    return nothing
end
@inline function (J::jacobian_combined{T1,T2,T3})(t::Float64,Y::T,YP::T,γ::Float64,p,run) where {T<:Vector{Float64},T1<:Function,T2,T3<:typeof(scalar_jacobian!)}
    J.base_func(J.J_base,t,Y,YP,γ,J.θ_tot)
    scalar_jacobian!(J.J_scalar,t,Y,YP,γ,p,run)
    return nothing
end
@inline function (J::jacobian_combined{T1,T2,T3})(J_new::SparseMatrixCSC{Float64,Int64},t::Float64,Y::T,YP::T,γ::Float64,p,run) where {T<:Vector{Float64},T1<:Function,T2,T3<:Function}
    J.base_func(J.J_base,t,Y,YP,γ,J.θ_tot)
    J.scalar_func(J.J_scalar,t,Y,YP,γ,J.θ_tot)
    @inbounds J_new.nzval .= J.sp.nzval
    return nothing
end
@inline function (J::jacobian_combined{T1,T2,T3})(J_new::SparseMatrixCSC{Float64,Int64},t::Float64,Y::T,YP::T,γ::Float64,p,run) where {T<:Vector{Float64},T1<:Function,T2,T3<:typeof(scalar_jacobian!)}
    J.base_func(J.J_base,t,Y,YP,γ,J.θ_tot)
    scalar_jacobian!(J.J_scalar,t,Y,YP,γ,p,run)
    @inbounds J_new.nzval .= J.sp.nzval
    return nothing
end

"""
method_functions definitions
"""
(f::method_functions)(::run_constant{method})      where method<:AbstractMethod                  = f.Dict_constant[method]
(f::method_functions)(::run_function{method,func}) where {method<:AbstractMethod,func<:Function} = f.Dict_function[method][func]
(f::method_functions)(::run_residual{func})        where func<:Function                          = f.Dict_residual[func]

Base.haskey(f::method_functions,::run_constant{method})      where method<:AbstractMethod                  = haskey(f.Dict_constant,method)
Base.haskey(f::method_functions,::run_function{method,func}) where {method<:AbstractMethod,func<:Function} = haskey(f.Dict_function,method) && haskey(f.Dict_function[method],func)
Base.haskey(f::method_functions,::run_residual{func})        where func<:Function                          = haskey(f.Dict_residual,func)