@inline function set_vars!(sol::R1, p::R2, Y::R3, YP::R3, t::R4, run::R5, opts::R6, bounds::R7;
    modify!::R8=set_var!,
    init_all::Bool=false,
    SOC::Float64 = (@inbounds sol.SOC[end])
    ) where {
        R1<:sol_output,
        R2<:model,
        R3<:Vector{Float64},
        R4<:Float64,
        R5<:AbstractRun,
        R6<:AbstractOptionsModel,
        R7<:boundary_stop_conditions,
        R8<:Function,
        }
    """
    Sets all the outputs for the sol. There are three kinds of variable outputs:
    
    1. `keep.x = true`: The variable is calculated and saved on every iteration
    
    2. `keep.x = false` WITHOUT the check `if keep.x ... end`: These variables  MUST be
        calculated to ensure that `check_simulation_stop!` works properly (e.g., check if
        `t > tf` or `SOC > SOC_max`), but they are not saved on every iteration
    
    3. `keep.x = false` WITH the check `if keep.x ... end`: These variables are not
        evaluated at all and may not even be calculable (e.g., `T` if there is no
        temperature enabled)
    """
    keep = opts.var_keep

    # these variables must be calculated, but they may not necessarily be kept
    modify!(sol.SOC, isempty(sol.SOC) ? SOC : calc_SOC(SOC, Y, t + run.t0, sol, p), (keep.SOC || init_all))
    modify!(sol.t,   t + run.t0, (keep.t || init_all))
    
    # these variables do not need to be calculated
    if keep.YP      modify!(sol.YP,      copy(YP)           ) end
    if keep.I       modify!(sol.I,       calc_I(Y, p)       ) end
    if keep.V       modify!(sol.V,       calc_V(Y, p)       ) end
    if keep.P       modify!(sol.P,       calc_P(Y, p)       ) end
    if keep.c_e     modify!(sol.c_e,     calc_c_e(Y, p)     ) end
    if keep.c_s_avg modify!(sol.c_s_avg, calc_c_s_avg(Y, p) ) end
    if keep.j       modify!(sol.j,       calc_j(Y, p)       ) end
    if keep.Φ_e     modify!(sol.Φ_e,     calc_Φ_e(Y, p)     ) end
    if keep.Φ_s     modify!(sol.Φ_s,     calc_Φ_s(Y, p)     ) end
    
    # exist as an optional output if the sol uses them
    if ( p.numerics.temperature === true           && keep.T    ) modify!(sol.T,    calc_T(Y,p)    ) end
    if ( p.numerics.aging === :SEI                 && keep.film ) modify!(sol.film, calc_film(Y,p) ) end
    if ( p.numerics.aging === :SEI                 && keep.SOH  ) modify!(sol.SOH,  calc_SOH(Y, p) ) end
    if ( !(p.numerics.aging === false)             && keep.j_s  ) modify!(sol.j_s,  calc_j_s(Y,p)  ) end
    if ( p.numerics.solid_diffusion === :quadratic && keep.Q    ) modify!(sol.Q,    calc_Q(Y,p)    ) end

    return nothing
end

@inline set_var!(x, x_val) = push!(x, x_val)
@inline function set_var!(x::T1, x_val::T2, keep::Bool) where {T1<:Vector{Float64},T2<:Float64}
    keep ? push!(x, x_val) : (@inbounds x[1] = x_val)
end
@inline function set_var!(x::T1, x_val::T2, keep::Bool) where {T1<:VectorOfArray{Float64,2,Array{Array{Float64,1},1}},T2<:AbstractVector{Float64}}
    keep ? push!(x, x_val) : (@inbounds x[1] .= x_val)
end

@inline function set_var_last!(x::T1, x_val::T2, keep=true) where {T1<:Vector{Float64},T2<:Float64}
    @inbounds x[end] = x_val
end
@inline function set_var_last!(x::T1, x_val::T2, keep=true) where {T1<:VectorOfArray{Float64,2,Array{Array{Float64,1},1}},T2<:AbstractVector{Float64}}
    @inbounds x[end] .= x_val
end

@inline function (sol::sol_output)(tspan::Union{Number,AbstractVector}; interp_bc::Symbol=:interpolate, k::Int64=1,kw...)
    if tspan isa UnitRange
        t = collect(tspan)
    elseif tspan isa Number
        t = Float64[Float64(tspan)]
    else
        t = tspan
    end

    var_keep = @inbounds @views sol.results[end].opts.var_keep
    function f(field)
        x = getproperty(sol, field)
        if field === :t
            return t
        elseif x isa AbstractArray{Float64} && getproperty(var_keep, field) && length(x) > 1
            return interpolate_variable(x, sol, t, interp_bc; k=k, kw...)
        else
            return x
        end
    end
    
    states_tot = @inbounds (f(field) for field in fieldnames(sol_output))

    sol = sol_output(states_tot...)

    return sol
end
@inline interpolate_variable(x::Any,y...;kw...) = x
@inline function interpolate_variable(x::R1, sol::R2, tspan::T1, interp_bc::Symbol;kw...) where {R1<:AbstractVector{Float64},R2<:sol_output,T1<:Union{Real,AbstractArray}}
    spl = Spline1D(sol.t, x; bc = (interp_bc == :interpolate ? "nearest" : (interp_bc == :extrapolate ? "extrapolate" : error("Invalid interp_bc method."))),kw...)
    out = spl(tspan)
    
    return out
end
@inline function interpolate_variable(x::R1,y...;kw...) where {R1<:Union{VectorOfArray{Float64,2,Array{Array{Float64,1},1}},Vector{Vector{Float64}}}}
    out = @inbounds @views hcat([interpolate_variable(x[i,:],y...;kw...) for i in 1:size(x,1)]...)

    return @inbounds VectorOfArray([out[i,:] for i in 1:size(out,1)])
end