@inline function set_vars!(model::R1, p::R2, Y::R3, YP::R3, t::R4, run::R5, opts::R6; init_all::Bool=false, modify!::Function=set_var!) where {R1<:model_output, R2<:param, R3<:Vector{Float64}, R4<:Float64, R5<:AbstractRun, R6<:options_model}

    ind  = p.ind
    keep = opts.var_keep

    # these variables must be calculated, but they may not necessarily be kept
    modify!(model.t,        (keep.t       || init_all), t + run.t0 )
    modify!(model.V,        (keep.V       || init_all), calc_V(Y, p, run) )
    modify!(model.I,        (keep.I       || init_all), calc_I(Y, model, run, p) )
    modify!(model.P,        (keep.P       || init_all), calc_P(Y, model, run, p) )
    modify!(model.c_s_avg,  (keep.c_s_avg || init_all), @views @inbounds Y[ind.c_s_avg] )
    modify!(model.SOC,      (keep.SOC     || init_all), @views @inbounds calc_SOC(model.c_s_avg[end], p) )
    
    # these variables do not need to be calculated
    if keep.YP  modify!(model.YP,  (keep.YP  || init_all), (keep.YP ? copy(YP) : YP)   ) end
    if keep.c_e modify!(model.c_e, (keep.c_e || init_all), @views @inbounds Y[ind.c_e] ) end
    if keep.j   modify!(model.j,   (keep.j   || init_all), @views @inbounds Y[ind.j]   ) end
    if keep.Φ_e modify!(model.Φ_e, (keep.Φ_e || init_all), @views @inbounds Y[ind.Φ_e] ) end
    if keep.Φ_s modify!(model.Φ_s, (keep.Φ_s || init_all), @views @inbounds Y[ind.Φ_s] ) end
    
    # exist as an optional output if the model uses them
    if ( p.numerics.temperature                    && keep.T )    modify!(model.T,    (keep.T    || init_all), @views @inbounds Y[ind.T]    ) end
    if ( p.numerics.aging === :SEI                 && keep.film ) modify!(model.film, (keep.film || init_all), @views @inbounds Y[ind.film] ) end
    if ( !(p.numerics.aging === false)             && keep.j_s )  modify!(model.j_s,  (keep.j_s  || init_all), @views @inbounds Y[ind.j_s]  ) end
    if ( p.numerics.solid_diffusion === :quadratic && keep.Q )    modify!(model.Q,    (keep.Q    || init_all), @views @inbounds Y[ind.Q]    ) end

    return nothing
end

@inline function set_var!(x::T1, append::Bool, x_val::T2) where {T1<:Vector{Float64},T2<:Float64}
    if append
        push!(x, x_val)
    else
        @inbounds x[1] = x_val
    end
end
@inline function set_var!(x::T1, append::Bool, x_val::T2) where {T1<:VectorOfArray{Float64,2,Array{Array{Float64,1},1}},T2<:AbstractVector}
    if append
        push!(x, x_val)
    else
        @inbounds x[1] .= x_val
    end
end

@inline function set_var_last!(x::T1, append::Bool, x_val::T2) where {T1<:Vector{Float64},T2<:Float64}
    @inbounds x[end] = x_val
end
@inline function set_var_last!(x::T1, append::Bool, x_val::T2) where {T1<:VectorOfArray{Float64,2,Array{Array{Float64,1},1}},T2<:AbstractVector}
    @inbounds x[end] .= x_val
end

@inline function interpolate_model(model::R1, tspan::T1, interp_bc::Symbol) where {R1<:model_output,T1<:Union{Number,AbstractVector}}
    dummy = similar(model.t)

    if tspan isa UnitRange
        t = collect(tspan)
    elseif tspan isa Real
        t = Float64[tspan]
    else
        t = tspan
    end

    f(x) = interpolate_variable(x, model, t, dummy, interp_bc)
    
    # collect all the variables for interpolation
    states_tot = Any[]
    @inbounds for field in fieldnames(model_output)
        if field === :t
            push!(states_tot, tspan)
        else
            x = getproperty(model, field)
            if x isa AbstractArray{Float64} && length(x) > 1
                push!(states_tot, f(x))
            else
                push!(states_tot, x)
            end
        end
        
    end

    model = R1(states_tot...)

    return model
end
@inline function interpolate_variable(x::R1, model::R2, tspan::T1, dummy::Vector{Float64}, interp_bc::Symbol) where {R1<:Vector{Float64},R2<:model_output,T1<:Union{Real,AbstractArray}}
    spl = Dierckx.Spline1D(model.t, x; bc = (interp_bc == :interpolate ? "nearest" : (interp_bc == :extrapolate ? "extrapolate" : error("Invalid interp_bc method."))))
    out = spl(tspan)
    
    return out
end

@inline function interpolate_variable(x::R1, model::R2, tspan::T1, dummy::Vector{Float64}, interp_bc::Symbol) where {R1<:Union{VectorOfArray{Float64,2,Array{Array{Float64,1},1}},Vector{Vector{Float64}}},R2<:model_output,T1<:Union{Real,AbstractArray}}
    @inbounds out = [copy(x[1]) for _ in tspan]

    @inbounds for i in eachindex(x[1])

        @inbounds for j in eachindex(x)
            @inbounds dummy[j] = x[j][i]
        end

        spl = Dierckx.Spline1D(model.t, dummy; bc = (interp_bc == :interpolate ? "nearest" : (interp_bc == :extrapolate ? "extrapolate" : error("Invalid interp_bc method."))))

        @inbounds for (j,t) in enumerate(tspan)
            @inbounds out[j][i] = spl(t)
        end

    end

    return VectorOfArray(out)
end