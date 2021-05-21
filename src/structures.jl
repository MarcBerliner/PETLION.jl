const const_Faradays  = 96485.3365
const const_Ideal_Gas = 8.31446261815324

@with_kw mutable struct run_info
    exit_reason::String = ""
    flag::Int64 = -1
    iterations::Int64 = -1
end

abstract type AbstractRun end

struct run_constant <: AbstractRun
    value::Float64
    method::Symbol
    t0::Float64
    tf::Float64
    I1C::Float64
    info::run_info
end

struct run_function <: AbstractRun
    func::Function
    value::Vector{Float64}
    method::Symbol
    t0::Float64
    tf::Float64
    I1C::Float64
    info::run_info
end
@inline value(run::run_constant) = run.value
@inline value(run::run_function) = @inbounds run.value[1]

@with_kw struct index_state <: AbstractUnitRange{Int64}
    start::Int64 = 0
    stop::Int64 = 0
    a::UnitRange{Int64} = 0:0
    p::UnitRange{Int64} = 0:0
    s::UnitRange{Int64} = 0:0
    n::UnitRange{Int64} = 0:0
    z::UnitRange{Int64} = 0:0
    sections::Tuple = ()
    var_type::Symbol = :NA
end

abstract type AbstractJacobian end
struct jacobian_symbolic <: AbstractJacobian
    func::Function
    sp::SparseMatrixCSC{Float64,Int64}
end
(jac::jacobian_symbolic)(x...) = jac.func(x...)

struct res_FD
    f!::Function
    YP_cache::Vector{Float64}
    θ_tot::Vector{Float64}
end
(res_FD::res_FD)(res, u) = res_FD.f!(res, u, res_FD.YP_cache, res_FD.θ_tot)

struct jacobian_AD <:AbstractJacobian
    f!::res_FD
    sp::SparseMatrixCSC{Float64,Int64}
    jac_cache::SparseDiffTools.ForwardColorJacCache
end
@inline function (jac::jacobian_AD)(J::R1, u::R2, x...) where {R1<:SparseMatrixCSC{Float64,Int64}, R2<:Vector{Float64}}
    forwarddiff_color_jacobian!(J, jac.f!, u, jac.jac_cache)
end

struct init_newtons_method{T}
    f_alg!::Function
    J_y_alg!::T
    f_diff!::Function
    Y0_alg::Vector{Float64}
    Y0_alg_prev::Vector{Float64}
    Y0_diff::Vector{Float64}
    res::Vector{Float64}
end

struct functions_model{T<:AbstractJacobian}
    f!::Function
    initial_guess!::Function
    J_y!::T
    initial_conditions::init_newtons_method{T}
    update_θ!::RuntimeFn
    int::Vector{Sundials.IDAIntegrator}
end

@with_kw mutable struct boundary_stop_conditions{T1<:Number,T2<:Float64}
    V_max::T1 = -1.0
    V_min::T1 = -1.0
    SOC_max::T1 = -1.0
    SOC_min::T1 = -1.0
    T_max::T1 = -1.0
    c_s_n_max::T1 = -1.0
    I_max::T1 = NaN
    I_min::T1 = NaN
    t_final_interp_frac::T2 = +1.0
    V_prev::T2 = -1.0
    SOC_prev::T2 = -1.0
    T_prev::T2 = -1.0
    c_s_n_prev::T2 = -1.0
    I_prev::T2 = -1.0
end

@inline function boundary_stop_conditions(V_max::Number, V_min::Number, SOC_max::Number, SOC_min::Number, T_max::Number, c_s_n_max::Number, I_max::Number, I_min::Number)
    boundary_stop_conditions(
        Float64(V_max),
        Float64(V_min),
        Float64(SOC_max),
        Float64(SOC_min),
        Float64(T_max),
        Float64(c_s_n_max),
        Float64(I_max),
        Float64(I_min),
        +1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
end

struct _discretizations_per_section
    p::Int64
    s::Int64
    n::Int64
    a::Int64
    z::Int64
    r_p::Int64
    r_n::Int64
end
struct discretizations_per_section
    p::Int64
    s::Int64
    n::Int64
    a::Int64
    z::Int64
    r_p::Int64
    r_n::Int64
    diff::Int64
    alg::Int64
    tot::Int64
end

@with_kw mutable struct _funcs_numerical
    rxn_p::Function = emptyfunc
    rxn_n::Function = emptyfunc
    OCV_p::Function = emptyfunc
    OCV_n::Function = emptyfunc
    D_s_eff::Function = emptyfunc
    D_eff::Function = emptyfunc
    K_eff::Function = emptyfunc
end

struct options_numerical
    cathode::Function
    anode::Function
    rxn_p::Function
    rxn_n::Function
    OCV_p::Function
    OCV_n::Function
    D_s_eff::Function
    rxn_rate::Function
    D_eff::Function
    K_eff::Function
    temperature::Bool
    solid_diffusion::Symbol
    Fickian_method::Symbol
    aging::Union{Bool,Symbol}
    edge_values::Symbol
    jacobian::Symbol
end

states_logic = model_states{
    Bool,
    Bool,
    Tuple
}

indices_states = model_states{
    index_state,
    index_state,
    Nothing,
}


@with_kw mutable struct options_model
    outputs::Tuple = (:t, :V)
    SOC::Number = 1.0
    abstol::Float64 = 1e-6
    reltol::Float64 = 1e-6
    abstol_init::Float64 = abstol
    reltol_init::Float64 = reltol
    maxiters::Int64 = 10_000
    check_bounds::Bool = true
    reinit::Bool = true
    verbose::Bool = false
    interp_final::Bool = true
    tstops::Vector{Float64} = Float64[]
    tdiscon::Vector{Float64} = Float64[]
    interp_bc::Symbol = :interpolate
    warm_start::Bool = false
    var_keep::states_logic = model_states_logic()
end

struct warm_start_info
    method::Symbol
    SOC::Float64 # rounded to 3rd decimal place
    I::Float64 # rounded to 3rd decimal place
end

struct cache_run
    θ_tot::ImmutableDict{Symbol,Vector{Float64}}
    θ_keys::ImmutableDict{Symbol,Vector{Symbol}}
    cache_name::String
    state_labels::Vector{Symbol}
    vars::Tuple
    outputs_tot::Tuple
    warm_start_dict::Dict{warm_start_info,Vector{Float64}}
    Y0::Vector{Float64}
    YP0::Vector{Float64}
    id::Vector{Int64}
    constraints::Vector{Int64}
end

@inline model_states_logic(outputs, cache::cache_run) = model_states_logic(outputs, cache.outputs_tot)

struct run_results{T<:AbstractRun}
    run::T
    tspan::Tuple{Float64,Float64}
    info::run_info
    run_index::UnitRange{Int64}
    int::Sundials.IDAIntegrator
    opts::options_model
    bounds::boundary_stop_conditions
    N::discretizations_per_section
    numerics::options_numerical
end

model_output = model_states{
    Array{Float64,1},
    VectorOfArray{Float64,2,Array{Array{Float64,1},1}},
    Array{run_results,1},
}
(model::model_output)(t::Union{Number,AbstractVector}; interp_bc::Symbol=:interpolate) = interpolate_model(model, t, interp_bc)
Base.length(model::model_output) = sum(result.info.iterations for result in model.results)
Base.isempty(model::model_output) = isempty(model.results)

abstract type AbstractParam end

struct param{T<:AbstractJacobian} <: AbstractParam
    θ::Dict{Symbol,Union{Float64,Vector{Float64}}}
    numerics::options_numerical
    N::discretizations_per_section
    ind::indices_states
    opts::options_model
    bounds::boundary_stop_conditions
    cache::cache_run
    funcs::ImmutableDict{Symbol,functions_model{T}}
    model_methods::Tuple
end

struct param_no_funcs <: AbstractParam
    θ::Dict{Symbol,Any}
    numerics::options_numerical
    N::discretizations_per_section
    ind::indices_states
    opts::options_model
    bounds::boundary_stop_conditions
    cache::cache_run
end


## Modifying Base functions
function Plots.plot(model::model_output, x_name::Symbol=:V; legend=false, ylabel=x_name, kwargs...)
    x = getproperty(model, x_name)
    
    !(size(x,2) === 1) ? x = x' : nothing
    if length(model.t) ≠ length(x) error("$x_name is not in `outputs`") end

    if model.t[end] ≥ 3600
        time_scale = 3600.0
        time_unit = "hr"
    else
        time_scale = 1.0
        time_unit = "s"
    end
    
    Plots.plot(model.t./time_scale, x;
        legend = legend,
        ylabel = x_name,
        xlabel = haskey(kwargs, :xlabel) ? kwargs[:xlabel] : "t ($(time_unit))",
        kwargs...
    )

end

function C_rate_string(I::Number;digits::Int64=4)
    I_rat = rationalize(Float64(I))
    num = I_rat.num
    den = I_rat.den

    if den > 100
        return "$(round(I;digits=digits))C"
    end
    
    str = I < 0 ? "-" : ""
    
    if isone(abs(num)) && !isone(abs(den))
        str *= "C"
    else
        str *= "$(abs(num))C"
    end
    
    if !isone(den)
        str *= "/$den"
    end
    
    return str
end

function Base.show(io::IO, p::AbstractParam)
    
    function show_bounds(title, min, max, units="")
        if isnan(min) && isnan(max)
            return ""
        end
        
        pad = 19
        str = "  $title "
        if isnan(max)
            str = rpad(str * "min:",pad)*"$min$units\n"
        elseif isnan(min)
            str = rpad(str * "max:",pad)*"$max$units\n"
        else
            str = rpad(str * "bounds:",pad)*"[$min$units, $max$units]\n"
        end
        
        return str
        
    end
    
    sp = p.numerics.solid_diffusion === :Fickian ? "  " : ""
    str = string(
    "$(replace(string(typeof(p)), "PETLION."=>"")):\n",
    "  Cathode: $(p.numerics.cathode), $(p.numerics.rxn_p), & $(p.numerics.OCV_p)\n",
    "  Anode:   $(p.numerics.anode), $(p.numerics.rxn_n), & $(p.numerics.OCV_n)\n",
    "  System:  $(p.numerics.D_s_eff), $(p.numerics.rxn_rate), $(p.numerics.D_eff), & $(p.numerics.K_eff)\n",
    :model_methods ∈ fieldnames(typeof(p)) && !isempty(p.model_methods) ? 
    "  Methods: $(join(p.model_methods, ", "))\n" : "",
    "  --------\n",
    "  Temperature:     $(p.numerics.temperature)\n",
    "  Solid diffusion: $(p.numerics.solid_diffusion)",
    p.numerics.solid_diffusion === :Fickian ? 
    ", $(p.numerics.Fickian_method)\n" : "\n",
    "  Aging:           $(p.numerics.aging)\n",
    "  Edge values:     $(p.numerics.edge_values)\n" ,
    show_bounds("Voltage", p.bounds.V_min, p.bounds.V_max, " V"),
    show_bounds("SOC", p.bounds.SOC_min, p.bounds.SOC_max),
    show_bounds("Current", p.bounds.I_min, p.bounds.I_max, "C"),
    p.numerics.temperature ?
    show_bounds("Temperature", NaN, p.bounds.T_max-273.15, " °C") : "",
    show_bounds("Anode sat.", NaN, p.bounds.c_s_n_max),
    "  --------\n",
    p.numerics.temperature ?
    "  N.a: $sp$(p.N.a)\n" : "",
    "  N.p: $sp$(p.N.p)\n",
    p.numerics.solid_diffusion === :Fickian ?
    "  N.r_p: $(p.N.r_p)\n" : "",
    "  N.s: $sp$(p.N.s)\n",
    "  N.n: $sp$(p.N.n)\n",
    p.numerics.solid_diffusion === :Fickian ?
    "  N.r_n: $(p.N.r_n)\n" : "",
    p.numerics.temperature ?
    "  N.z: $sp$(p.N.z)\n" : "",
    )
    
    print(io, str[1:end-1])
end

function Base.show(io::IO, ind::indices_states)
    
    outputs_tot = Symbol[]
    @inbounds for (name,_type) in zip(fieldnames(typeof(ind)), fieldtypes(typeof(ind)))
        if _type === index_state
            push!(outputs_tot, name)
        end
    end
    outputs_tot = (outputs_tot...,)

    vars = Symbol[]
    tot = Int64[]
    indices = UnitRange{Int64}[]

    for field in outputs_tot
        ind_var = getproperty(ind, field)
        if ind_var.var_type ∈ (:differential, :algebraic)
            push!(vars, field)
            push!(tot, ind_var.start)
            push!(indices, ind_var.start:ind_var.stop)
        end
    end
    vars    .= vars[sortperm(tot)]
    indices .= indices[sortperm(tot)]

    pad = maximum(length.(String.(vars)))+2

    str = "indices_states:\n"
    for (i,var) in enumerate(vars)
        index = indices[i]
        str *= "  " * rpad("$(var): ", pad)
        str *= "$(length(index) > 1 ? index : index[1])\n"
    end

    print(io, str[1:end-1])
end

function Base.show(io::IO, result::run_results{T}) where {T<:AbstractRun}
    
    fix(x, digits=2) = round(x, digits=digits)
    run = result.run
    str = "Results for $(run.method) "
    if T === run_constant
        if     run.method === :I
            str *= "= $(C_rate_string(value(run)/run.I1C;digits=2))"
        elseif run.method === :V
            str *= "= $(fix(value(run))) V"
        elseif run.method === :P
            str *= "= $(fix(value(run))) W"
        end
    elseif T === run_function
        str *= "function"
    end
    str *= " from ($(fix(result.tspan[1])) s, $(fix(result.tspan[2])) s)"

    print(io, str)
end

@inbounds @views function Base.show(io::IO, run::T) where {T<:AbstractRun}
    str = "Run for $(run.method) "
    fix(x,digits=2) = round(x, digits=2)
    if T === run_constant
        if     run.method === :I
            str *= "= $(C_rate_string(value(run)/run.I1C;digits=2))"
        elseif run.method === :V
            str *= "= $(fix(value(run))) V"
        elseif run.method === :P
            str *= "= $(fix(value(run))) W"
        end
    elseif T === run_function
        str *= "function"
    end
    if !isempty(run.info.exit_reason)
        str *= " with exit: $(run.info.exit_reason)"
    end
    
    print(io, str)
end

@inbounds @views function Base.show(io::IO, model::model_output)
    results = model.results
    function str_runs()
    
        str = length(results) === 1 ? "  Run: " : "  Runs:"
        str *= " "^4

        methods = string.([result.run.method for result in results])
        @inbounds for (i,result) in enumerate(results)
            if result.run isa run_function
                    methods[i] *= "func"
            end
        end

        counts = ones(Int64, length(methods))

        @inbounds for i in length(counts)-1:-1:1
            if methods[i] === methods[i+1]
                counts[i] += counts[i+1]
                deleteat!(methods, i+1)
                deleteat!(counts, i+1)
            end
        end

        @inbounds for i in 1:length(methods)
            methods[i] = (counts[i] === 1 ? methods[i] : "$(counts[i]) $(methods[i])")
        end
        str *= join(methods, " → ") * "\n"

        return str
    end
    
    title = "PETLION model"
    if length(model.results) ≥ 1
        str = @views @inbounds string(
                "$title\n",
                "  --------\n",
                str_runs(),
                "  Time:    $(round(model.t[end];   digits = 2)) s\n",
                "  Voltage: $(round(model.V[end];   digits = 4)) V\n",
                "  Current: $(C_rate_string(model.I[end];digits=4))\n",
                "  Power:   $(round(model.P[end];   digits = 4)) W\n",
                "  SOC:     $(round(model.SOC[end]; digits = 4))\n",
                length(model.T) > 0 ? 
                "  Temp.:   $(round(maximum(model.T[end])-273.15; digits = 4)) °C\n"
                : "",
                "  Exit:    $(model.results[end].info.exit_reason)",
            )
    else
        str = "$title: empty"
    end
    
    print(io, str)
end

@inbounds @views function Base.show(io::IO, ind::states_logic)
    
    str = "$(typeof(ind)) using:\n"
    @inbounds for field in fieldnames(typeof(ind))
        x = getproperty(ind,field)
        if x isa Bool && x
            str *= "  $field\n"
        end
    end
    
    print(io, str[1:end-1])
end

function _MTK_MatVecProd(A, x; simple::Bool = true)
    """
    Change matrix-vector multiplication in ModelingToolkit to ignore 0s in the matrix.
    This can speed up computation time without resorting to using the `simplify` function.
    """
    n, m = size(A)

    b = zeros(Num, n)
    count = zeros(Int, n)

    @inbounds for ind_i in 1:n, ind_j in 1:m

        @inbounds @views A_val = A[ind_i,ind_j]
        A_simple = simple ? simplify(A_val) : A_val

        if !isequal(A_simple, 0)

            if @inbounds @views count[ind_i] == 0
                @inbounds @views b[ind_i] = A_val*x[ind_j]
                @inbounds @views count[ind_i] += 1
            else
                @inbounds @views b[ind_i] += A_val*x[ind_j]
            end

        end
    end

    return b
end
# overloads of * to use _MTK_MatVecProd when appropriate
Base.:*(A::AbstractMatrix, x::AbstractVector{Num}; simple::Bool = true) = _MTK_MatVecProd(A, x; simple=simple)
Base.:*(A::Union{Array{T,2}, Array{T,2}, Array{T,2}, Array{T,2}}, x::StridedArray{Num, 1}; simple::Bool = true) where {T<:Union{Float32, Float64}} = _MTK_MatVecProd(A, x; simple=simple)

Base.deleteat!(a::VectorOfArray, i::Integer) = (Base._deleteat!(a.u, i, 1); a.u)

function emptyfunc end