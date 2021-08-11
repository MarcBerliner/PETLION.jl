@inline calc_V(Y::Vector{<:Number}, p::AbstractParam) = @inbounds Y[p.ind.Φ_s[1]] - Y[p.ind.Φ_s[end]]
@inline calc_I(Y::Vector{<:Number}, p::AbstractParam) = @inbounds Y[end]
@inline calc_P(Y::Vector{<:Number}, p::AbstractParam) = calc_I(Y,p)*p.θ[:I1C]*calc_V(Y,p)

@inline method_I(Y, p)   = calc_I(Y,p)
@inline method_V(Y, p)   = calc_V(Y,p)
@inline method_P(Y, p)   = calc_P(Y,p)
@inline method_η_p(Y, p) = calc_η_plating(Y,p)
@inline method_res(Y, p) = 0.0

@inline scalar_residual!(res::Vector{T},t,Y,YP,p,run::run_constant{method,<:Any}) where {method<:AbstractMethod,T<:Number} = @inbounds (res[end] = method(Y,p) - value(run))

@inline scalar_residual!(res::Vector{T},t,Y,YP,p,run::run_residual{method_res,func}) where {T<:Number,func<:Function} = @inbounds (res[end] = p.θ[:_residual_val] - run.func(t,Y,YP,p))

@inline scalar_residual!(res::Vector{T},t,Y,YP,p,run::run_function{method,func}) where {method<:AbstractMethod,T<:Num,func<:Function}     = @inbounds (res[end] = method(Y,p) - run.func(t,Y,YP,p))
@inline scalar_residual!(res::Vector{T},t,Y,YP,p,run::run_function{method,func}) where {method<:AbstractMethod,T<:Float64,func<:Function} = @inbounds (val = run.func(t,Y,YP,p); run.value .= val; res[end] = method(Y,p) - val)

@inline function scalar_jacobian(t,Y,YP,γ,p::AbstractParam,run::T) where T<:Union{run_constant,run_function}
    J = get_jacobian_sparsity(p,run;jac_type=typeof(t))
    J_nzval = @views @inbounds J.nzval[collect(1:length(J.nzval))]
    scalar_jacobian!(J_nzval,t,Y,YP,γ,p,run)
    return J
end
@inline @inbounds function scalar_jacobian!(J::SM,::T,Y::T2,::T2,γ::T,p::AbstractParam,::AbstractRun{method_I,<:Any}) where {T<:Number,SM<:SubArray{T,1,Vector{T},Tuple{Vector{Int64}},false},T2<:Vector{T}}
    J[1] = 1.0
    return nothing
end
@inline @inbounds function scalar_jacobian!(J::SM,::T,Y::T2,::T2,γ::T,p::AbstractParam,::AbstractRun{method_V,<:Any}) where {T<:Number,SM<:SubArray{T,1,Vector{T},Tuple{Vector{Int64}},false},T2<:Vector{T}}
    J[1] = 1.0
    J[2] = -1.0
    return nothing
end
@inline @inbounds function scalar_jacobian!(J::SM,::T,Y::T2,::T2,γ::T,p::AbstractParam,::AbstractRun{method_P,<:Any}) where {T<:Number,SM<:SubArray{T,1,Vector{T},Tuple{Vector{Int64}},false},T2<:Vector{T}}
    I1C = p.θ[:I1C]
    I = calc_I(Y,p)*I1C
    V = calc_V(Y,p)
    J[1] = I
    J[2] = -I
    J[3] = V*I1C
    return nothing
end
@inline @inbounds function scalar_jacobian!(J::SM,::T,Y::T2,::T2,γ::T,p::AbstractParam,::AbstractRun{method_η_p,<:Any}) where {T<:Number,SM<:SubArray{T,1,Vector{T},Tuple{Vector{Int64}},false},T2<:Vector{T}}
    J[1] = -1
    J[2] = 1
    return nothing
end

@inbounds function get_jacobian_sparsity(p::AbstractParam, ::AbstractRun{method_I,<:Any};jac_type::DataType=Float64)
    J = spzeros(jac_type,p.N.tot)
    J[p.ind.I] .= 1

    return J
end
@inbounds function get_jacobian_sparsity(p::AbstractParam, ::AbstractRun{method_V,<:Any};jac_type::DataType=Float64)
    J = spzeros(jac_type,p.N.tot)
    J[p.ind.Φ_s[[1,end]]] .= 1

    return J
end
@inbounds function get_jacobian_sparsity(p::AbstractParam, ::AbstractRun{method_P,<:Any};jac_type::DataType=Float64)
    J = spzeros(jac_type,p.N.tot)
    J[p.ind.I] .= 1
    J[p.ind.Φ_s[[1,end]]] .= 1

    return J
end
@inbounds function get_jacobian_sparsity(p::AbstractParam, ::AbstractRun{method_η_p,<:Any};jac_type::DataType=Float64)
    J = spzeros(jac_type,p.N.tot)
    J[p.ind.Φ_e.n[1]] = 1
    J[p.ind.Φ_s.n[1]] = 1

    return J
end

@inline function numargs(f::T,maxargs::Int64=99) where T<:Function
    maxval = 0
    maxargs += 1
    @inbounds for m in methods(f)
        val = num_types_in_tuple(m.sig)
        if val === maxargs
            maxval = val
            break
        elseif val ≤ maxargs && val > maxval
            maxval = val
        end
    end
    return maxval - 1
end
@inline num_types_in_tuple(sig::DataType) = length(sig.parameters)
@inline num_types_in_tuple(sig::UnionAll) = length(Base.unwrap_unionall(sig).parameters)

@inline function redefine_func(f::T;
    args::Int64=numargs(f,4),
    ) where T<:Function
    
    if     args === 4
        f_new = f
        # inputs = (:t,:Y,:YP,:p)
    elseif args === 3
        f_new = (t,Y,YP,p) -> f(t,Y,p)
        # inputs = (:t,:Y,:p)
    elseif args === 2
        f_new = (t,Y,YP,p) -> f(t,p)
        # inputs = (:t,:p)
    elseif args === 1
        f_new = (t,Y,YP,p) -> f(t)
        # inputs = (:t)
    else
        error("Input function must have at least one argument.")
    end

    return f_new#, inputs
end

function _get_method_funcs(p::param, run::run_function)
    θ_sym, Y, YP, t, SOC, I, γ, p_sym, θ_keys = get_symbolic_vars(p)
    res = similar(Y) .= 0.0

    scalar_residual!(res,t,Y,YP,p_sym,run)
    Jac_constant = scalar_jacobian(t,Y,YP,γ,p_sym,run)

    J_Y = J_YP = J_vec = nothing
    is_differentiable = true
    try
        J_Y  = @inbounds Symbolics.sparsejacobian([res[end]], Y)[:]
        J_YP = @inbounds Symbolics.sparsejacobian([res[end]], YP)[:]
        J_vec = J_Y .+ γ.*J_YP
    catch
        is_differentiable = false
    end

    if is_differentiable && !(Jac_constant === J_vec)
        return differentiate_residual_func(p,run,J_vec,J_Y,J_YP,res,θ_sym,Y,YP,t,SOC,I,γ,p_sym,θ_keys)
    else
        return _get_method_funcs_no_differentiation(p,run)
    end
end

function _get_method_funcs(p::param, run::run_residual)
    θ_sym, Y, YP, t, SOC, I, γ, p_sym, θ_keys = get_symbolic_vars(p)
    res = similar(Y) .= 0.0

    scalar_residual!(res,t,Y,YP,p_sym,run)
    
    J_Y  = @inbounds Symbolics.sparsejacobian([res[end]], Y)[:]
    J_YP = @inbounds Symbolics.sparsejacobian([res[end]], YP)[:]
    J_vec = J_Y .+ γ.*J_YP

    return differentiate_residual_func(p,run,J_vec,J_Y,J_YP,res,θ_sym,Y,YP,t,SOC,I,γ,p_sym,θ_keys)
end

function differentiate_residual_func(p::param,run::T,J_vec,J_Y,J_YP,res,θ_sym,Y,YP,t,SOC,I,γ,p_sym,θ_keys) where T<:Union{run_function,run_residual}
    scalar_contains_Y_diff = @inbounds !isempty(J_Y[1:p.N.diff].nzval)
    scalar_contains_YP     = !isempty(J_YP.nzval)

    J_sp_scalar = spzeros(Float64,p.N.tot)
    @inbounds J_sp_scalar[J_vec.nzind] .= 1
    
    """
    Updating the theta vector to ensure any new parameters will be accounted for
    """
    θ_keys_scalar = @inbounds get_only_θ_used_in_model(θ_sym, θ_keys, res[end])[2]

    θ_keys = deepcopy(p.cache.θ_keys)
    @inbounds for key in θ_keys_scalar
        if !(key ∈ θ_keys)
            push!(θ_keys, key)
        end
    end

    θ_tot      = [p.θ[key] for key in θ_keys]
    θ_sym_slim = [p_sym.θ[key] for key in θ_keys]
    update_θ!(θ_tot,θ_keys,p.θ)

    J_scalar_func = eval(build_function(J_vec.nzval,t,Y,YP,γ,θ_sym_slim; expression=Val{false})[2])

    J_sp_base = p.funcs.J_y!.sp
    J_base_func = p.funcs.J_y!

    """
    Creating the scalar function
    """
    if scalar_contains_YP
        vars_in_residual = @inbounds Symbolics.get_variables(res[end])
        # Find all indices where YP is used in the residual equation
        ind_differential = findall(in(vars_in_residual), @views @inbounds YP[1:p.N.diff])
        
        @assert !isempty(ind_differential)

        res_algebraic = @inbounds res[end]

        residuals_PET!(res,t,Y,YP,p_sym)
        @inbounds for ind in ind_differential
            res_algebraic = substitute(res_algebraic, Dict(YP[ind] => res[ind] + YP[ind]))
        end

        scalar_residal_alg! = eval(build_function(res_algebraic,t,Y,YP,θ_sym_slim; expression=Val{false}))

        J_alg_vec = Symbolics.sparsejacobian([res_algebraic], Y[p.N.diff+1:end])[:]
        
        J_scalar_alg_func = eval(build_function(J_alg_vec.nzval,t,Y,YP,γ,θ_sym_slim; expression=Val{false})[2])
        
        J_sp_alg_scalar = spzeros(Float64,p.N.alg)
        J_sp_alg_scalar[J_alg_vec.nzind] .= 1
    else
        scalar_residal_alg! = scalar_residual!

        if scalar_contains_Y_diff
            J_scalar_alg_func = eval(build_function((@inbounds J_vec[p.N.diff+1:end].nzval),t,Y,YP,γ,θ_sym_slim; expression=Val{false})[2])
        else
            J_scalar_alg_func = J_scalar_func
        end
        J_sp_alg_scalar = @inbounds J_sp_scalar[p.N.diff+1:end]
    end

    return combine_Jac_and_res(p,J_sp_base,J_base_func,J_sp_scalar,J_scalar_func,θ_tot,θ_keys,scalar_residual!,scalar_residal_alg!,J_scalar_alg_func,J_sp_alg_scalar)
end

function _get_method_funcs_no_differentiation(p::param, run::AbstractRun)
    J_sp_scalar = get_jacobian_sparsity(p,run)
    θ_tot = p.cache.θ_tot
    θ_keys = p.cache.θ_keys
    update_θ!(θ_tot,θ_keys,p.θ)
    
    J_scalar_func = scalar_jacobian!
    J_sp_base = p.funcs.J_y!.sp
    J_base_func = p.funcs.J_y!

    scalar_residal_alg! = scalar_residual!
    J_scalar_alg_func = J_scalar_func
    J_sp_alg_scalar = J_sp_scalar[p.N.diff+1:end]
    
    return combine_Jac_and_res(p,J_sp_base,J_base_func,J_sp_scalar,J_scalar_func,θ_tot,θ_keys,scalar_residual!,scalar_residal_alg!,J_scalar_alg_func,J_sp_alg_scalar)
end

function combine_Jac_and_res(p,J_sp_base,J_base_func,J_sp_scalar,J_scalar_func,θ_tot,θ_keys,scalar_residual!,scalar_residal_alg!,J_scalar_alg_func,J_sp_alg_scalar)
    J_full = _get_jacobian_combined(
        J_sp_base,
        J_base_func,
        J_sp_scalar,
        J_scalar_func,
        θ_tot,
        θ_keys,
        )
    
    R_full = residual_combined(
        p.funcs.f_diff!,
        p.funcs.f_alg!,
        scalar_residual!,
        1:p.N.diff,
        p.N.diff+1:p.N.tot,
        θ_tot,
        θ_keys,
    )
    
    J_alg = _get_jacobian_combined(
        (@inbounds J_sp_base[p.N.diff+1:p.N.tot-1,p.N.diff+1:p.N.tot]),
        p.funcs.J_y_alg!,
        J_sp_alg_scalar,
        J_scalar_alg_func,
        θ_tot,
        θ_keys,
        )

    R_diff = residual_combined(
        p.funcs.f_diff!,
        emptyfunc,
        emptyfunc,
        1:p.N.diff,
        1:0,
        θ_tot,
        θ_keys,
    )
    
    R_alg = residual_combined(
        emptyfunc,
        p.funcs.f_alg!,
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

function get_method_funcs!(p::param,run::run_constant{method,<:Any}) where method<:AbstractMethod
    p.funcs.Dict_constant[method] = _get_method_funcs_no_differentiation(p,run)
end
function get_method_funcs!(p::param,run::run_function{method,func}) where {method<:AbstractMethod,func<:Function}
    if !haskey(p.funcs.Dict_function,method)
        p.funcs.Dict_function[method] = Dict{DataType,jacobian_combined}()
    end

    func_only_uses_time = true
    try
        # Seeing if the residual function uses Y or YP. If there's an error or if it does not output a number,
        # get a Jacobian which is differentiable. Otherwise, the Jacobian is known
        if !(run.func(0.0,nothing,nothing,p) isa Number)
            func_only_uses_time = false
        end
    catch
        func_only_uses_time = false
    end
        
    if func_only_uses_time
        p.funcs.Dict_function[method][func] = _get_method_funcs_no_differentiation(p,run)
    else
        p.funcs.Dict_function[method][func] = _get_method_funcs(p,run)
    end

    return nothing
end
function get_method_funcs!(p::param,run::run_residual{method_res,func}) where {func<:Function}
    p.funcs.Dict_residual[func] = _get_method_funcs(p,run)
end

"""
Multiple dispatch for the residuals function
"""
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:Function,T2<:Function,T3<:Function,T<:AbstractVector{Float64}}
    r.f_diff!((@views @inbounds res[r.ind_diff]), t, Y, YP, r.θ_tot)
    r.f_alg!( (@views @inbounds res[r.ind_alg]),  t, Y, YP, r.θ_tot)
    @inbounds res[end] = r.f_scalar!(t,Y,YP,r.θ_tot)
    return nothing
end
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:typeof(emptyfunc),T2<:Function,T3<:Function,T<:AbstractVector{Float64}}
    r.f_alg!(res,t,Y,YP,r.θ_tot)
    @inbounds res[end] = r.f_scalar!(t,Y,YP,r.θ_tot)
    return nothing
end
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:Function,T2<:Function,T3<:typeof(scalar_residual!),T<:AbstractVector{Float64}}
    r.f_diff!((@views @inbounds res[r.ind_diff]), t, Y, YP, r.θ_tot)
    r.f_alg!( (@views @inbounds res[r.ind_alg]),  t, Y, YP, r.θ_tot)
    scalar_residual!(res,t,Y,YP,p,run)
    return nothing
end
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:typeof(emptyfunc),T2<:Function,T3<:typeof(scalar_residual!),T<:AbstractVector{Float64}}
    r.f_alg!(res,t,Y,YP,r.θ_tot)
    scalar_residual!(res,t,Y,YP,p,run)
    return nothing
end
@inline function (r::residual_combined{T1,T2,T3})(res::T,t,Y,YP,p,run) where {T1<:Function,T2<:typeof(emptyfunc),T3<:typeof(emptyfunc),T<:AbstractVector{Float64}}
    r.f_diff!(res, t, Y, YP, r.θ_tot)
    return nothing
end

"""
Multiple dispatch for the Jacobian function
"""
@inline function (J::jacobian_combined)(J_new::SparseMatrixCSC{Float64,Int64},x...)
    J(x...)
    @inbounds J_new.nzval .= J.sp.nzval
    return nothing
end
@inline function (J::jacobian_combined{T1,T2,T3})(t,Y,YP,γ,p,run) where {T1<:Function,T2,T3<:Function}
    J.base_func(J.J_base,t,Y,YP,γ,J.θ_tot)
    J.scalar_func(J.J_scalar,t,Y,YP,γ,J.θ_tot)
    return nothing
end
@inline function (J::jacobian_combined{T1,T2,T3})(t,Y,YP,γ,p,run) where {T1<:Function,T2,T3<:typeof(scalar_jacobian!)}
    J.base_func(J.J_base,t,Y,YP,γ,J.θ_tot)
    scalar_jacobian!(J.J_scalar,t,Y,YP,γ,p,run)
    return nothing
end
@inline function (J::jacobian_combined{T1,T2,T3})(t,Y::AbstractVector{Float64},YP::AbstractVector{Float64},γ,p::param{<:jacobian_AD},run) where {T1<:Function,T2,T3<:Function}
    J.scalar_func(J.J_scalar,t,Y,YP,γ,J.θ_tot)
    res_FD = J.base_func.f!
    if size(J.sp) === (p.N.alg,p.N.alg)
        @inbounds @views res_FD.Y_cache[1:res_FD.N.diff] .= Y[1:res_FD.N.diff]
        Y_new = @views @inbounds Y[p.N.diff+1:end]
        @inbounds res_FD.YP_cache .= 0.0
    else
        @inbounds res_FD.YP_cache .= YP
        Y_new = Y
    end
    J.base_func(t,Y_new,YP,γ,p,run)
    J.J_base .= J.base_func.sp.nzval
    return nothing
end
@inline function (J::jacobian_combined{T1,T2,T3})(t,Y::AbstractVector{Float64},YP::AbstractVector{Float64},γ,p::param{<:jacobian_AD},run) where {T1<:Function,T2,T3<:typeof(scalar_jacobian!)}
    scalar_jacobian!(J.J_scalar,t,Y,YP,γ,p,run)
    res_FD = J.base_func.f!
    if size(J.sp) === (p.N.alg,p.N.alg)
        @inbounds @views res_FD.Y_cache[1:res_FD.N.diff] .= Y[1:res_FD.N.diff]
        Y_new = @views @inbounds Y[p.N.diff+1:end]
        @inbounds res_FD.YP_cache .= 0.0
    else
        @inbounds res_FD.YP_cache .= YP
        Y_new = Y
    end
    J.base_func(t,Y_new,YP,γ,p,run)
    J.J_base .= J.base_func.sp.nzval
    return nothing
end

"""
model_funcs definitions
"""
(f::model_funcs)(::run_constant{method,input}) where {method<:AbstractMethod,input<:Any}     = f.Dict_constant[method]
(f::model_funcs)(::run_function{method,func})  where {method<:AbstractMethod,func<:Function} = f.Dict_function[method][func]
(f::model_funcs)(::run_residual{method,func})  where {method<:method_res,func<:Function}     = f.Dict_residual[func]

Base.haskey(f::model_funcs,::run_constant{method,input})  where {method<:AbstractMethod,input<:Any}     = haskey(f.Dict_constant,method)
Base.haskey(f::model_funcs,::run_function{method,func})   where {method<:AbstractMethod,func<:Function} = haskey(f.Dict_function,method) && haskey(f.Dict_function[method],func)
Base.haskey(f::model_funcs,::run_residual{method,func})   where {method<:method_res,func<:Function}     = haskey(f.Dict_residual,func)