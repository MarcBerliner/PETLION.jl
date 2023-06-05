# Get all the states and outputs
begin 
    states = Symbol.(keys(merge(model_states_and_outputs()...)))
    
    for remove_state in (:SOC,)
        deleteat!(states, findall(states .== remove_state))
    end
    outputs_vec = Meta.parse.(["if keep.$state modify!(sol.$state, calc_$state(Y, p) ) end" for state in states])
end

eval(quote
@inline function set_vars!(sol::R1, p::R2, Y::R3, YP::R3, t::R4, run::R5, opts::R6, bounds::R7;
    modify!::R8=set_var!,
    init_all::Bool=false,
    SOC::Number = (@inbounds sol.SOC[end])
    ) where {
        R1<:solution,
        R2<:model,
        R3<:AbstractVector{<:Float64},
        R4<:Float64,
        R5<:AbstractRun,
        R6<:options_simulation_immutable,
        R7<:boundary_stop_conditions_immutable,
        R8<:Function,
        }
    """
    Sets all the outputs for the solution.
    """
    keep = opts.var_keep

    modify!(sol.SOC, isempty(sol.SOC) ? SOC : calc_SOC(SOC, Y, t + run.t0, sol, p), (keep.SOC || init_all))
    modify!(sol.t,   t + run.t0, (keep.t || init_all))
    
    # these variables do not need to be calculated
    if keep.YP modify!(sol.YP, copy(YP)) end
    $(outputs_vec...)

    return nothing
end
end)

@inline set_var!(x, x_val) = push!(x, x_val)
@inline function set_var!(x::T1, x_val::T2, keep::Bool) where {T1<:AbstractVector{<:Float64},T2<:Number}
    keep ? push!(x, x_val) : (@inbounds x[1] = x_val)
end
@inline function set_var!(x::T1, x_val::T2, keep::Bool) where {T1<:VectorOfArray{Float64,2,Array{Array{Float64,1},1}},T2<:AbstractVector{<:Float64}}
    keep ? push!(x, x_val) : (@inbounds x[1] .= x_val)
end

@inline function set_var_last!(x::T1, x_val::T2, keep=true) where {T1<:AbstractVector{<:Float64},T2<:Number}
    @inbounds x[end] = x_val
end
@inline function set_var_last!(x::T1, x_val::T2, keep=true) where {T1<:VectorOfArray{Float64,2,Array{Array{Float64,1},1}},T2<:AbstractVector{<:Float64}}
    @inbounds x[end] .= x_val
end

export reset_t!
@inline reset_t!(sol::solution) = (sol.t .-= sol.t[1]; sol)

@inline function tspan_index(t::T1,tspans::T2) where {T1<:Number,T2<:Vector{Tuple{Float64, Float64}}}
    if t < tspans[1][1]
        return 1
    end
    @inbounds for i in 1:length(tspans)
        t0,tf=tspans[i]
        if t0 ≤ t ≤ tf
            return i
        end
    end
    return length(tspans)
end
@inline Base.broadcasted(::typeof(tspan_index), t_vec::T, tspans) where T<:AbstractArray = [tspan_index(t,tspans) for t in t_vec]

(sol::solution)(t::T;kw...) where {T<:Number} = sol(Float64[t];kw...)
(sol::solution)(t::T;kw...) where {T<:AbstractUnitRange} = sol(collect(t);kw...)
@inline function (sol::solution)(t::T; interp_bc::Symbol=:interpolate, k::Int64=3,kw...) where T<:AbstractArray{<:Number}
    tspans = [sol.results[i].tspan for i in 1:length(sol)]
    ind_sol = tspan_index.(t,tspans)
    ind_sol_unique = unique(ind_sol)

    tspan_indices = Dict(ind_sol_unique .=> [findall(ind_sol .== ind) for ind in ind_sol_unique])
    sol_indices = Dict(ind_sol_unique .=> [sol.results[ind].run_index for ind in ind_sol_unique])

    if interp_bc == :interpolate
        bc = "nearest"
    elseif interp_bc == :extrapolate
        bc = "extrapolate"
    else
        error("Invalid interp_bc method.")
    end

    var_keep = @inbounds @views sol.results[end].opts.var_keep
    function f(field::Symbol)
        x = getproperty(sol, field)
        if field == :t
            return t
        elseif x isa AbstractArray{Float64} && getproperty(var_keep, field) && length(x) > 1
            return interpolate_variable(x, sol, t, interp_bc, tspan_indices, sol_indices; k=k, bc=bc, kw...)
        elseif field == :results
            return x[ind_sol_unique]
        else
            return x
        end
    end
    
    states_tot = @inbounds (f(field) for field in fieldnames(solution))
    
    sol_interp = solution(states_tot...)

    return sol_interp
end
@inline interpolate_variable(x::Any,y...;kw...) = x
@inline function interpolate_variable(x::R1, sol::R2, tspan::T1, interp_bc::Symbol,tspan_indices::Dict{Int64,Vector{Int64}},sol_indices::Dict{Int64,UnitRange{Int64}};k::Int64=3,kw...) where {R1<:AbstractVector{<:Float64},R2<:solution,T1<:AbstractArray{<:Number}}
    out = zeros(Float64,length(tspan))
    
    @inbounds for ind in keys(tspan_indices)
        tspan_ind = tspan_indices[ind]
        sol_inds = sol_indices[ind]
        
        spl = Spline1D((@views @inbounds sol.t[sol_inds]), (@views @inbounds x[sol_inds]);
            k=min(k,length(sol_inds)),
            kw...,
        )
        @inbounds out[tspan_ind] = spl((@views tspan[tspan_ind]))
    end
    
    return out
end
@inline function interpolate_variable(x::R1,y...;kw...) where {R1<:Union{VectorOfArray{Float64,2,Array{Array{Float64,1},1}},Vector{Vector{Float64}}}}
    out = @inbounds @views hcat([interpolate_variable(x[i,:],y...;kw...) for i in 1:size(x,1)]...)

    return @inbounds VectorOfArray([out[i,:] for i in 1:size(out,1)])
end