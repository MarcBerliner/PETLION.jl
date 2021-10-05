# https://en.wikipedia.org/wiki/2019_redefinition_of_the_SI_base_units
const const_Faradays  = 96485.3321233
const const_Ideal_Gas = 8.31446261815324

abstract type AbstractJacobian <: Function end
abstract type AbstractParam{T<:AbstractJacobian,temp,solid_diff,Fickian,age} end
abstract type AbstractMethod end
abstract type AbstractRun{T<:AbstractMethod,input<:Any} end

struct method_I   <: AbstractMethod end
struct method_V   <: AbstractMethod end
struct method_P   <: AbstractMethod end

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
discretizations_per_section(p,s,n,a,z,r_p,r_n) = discretizations_per_section(p,s,n,a,z,r_p,r_n,-1,-1,-1)

Base.@kwdef mutable struct run_info
    exit_reason::String = ""
    flag::Int64 = -1
    iterations::Int64 = -1
end

struct run_constant{T<:AbstractMethod,in<:Union{Number,Symbol,Function}} <: AbstractRun{T,in}
    input::in
    value::Vector{Float64}
    method::T
    t0::Float64
    tf::Float64
    name::String
    info::run_info
end
struct run_function{T<:AbstractMethod,func<:Function} <: AbstractRun{T,func}
    func::func
    value::Vector{Float64}
    method::T
    t0::Float64
    tf::Float64
    name::String
    info::run_info
end
@inline value(run::AbstractRun) = @inbounds run.value[1]

Base.@kwdef struct index_state <: AbstractUnitRange{Int64}
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

struct jacobian_symbolic{T<:Function} <: AbstractJacobian
    func::T
    sp::SparseMatrixCSC{Float64,Int64}
end
(jac::jacobian_symbolic{<:Function})(x...) = jac.func(x...)

struct res_FD{T<:Function} <: Function
    f!::T
    Y_cache::Vector{Float64}
    YP_cache::Vector{Float64}
    θ_tot::Vector{Float64}
    N::discretizations_per_section
end
function (res_FD::res_FD{T})(res::AbstractVector{<:Number}, Y::AbstractVector{<:Number}) where T<:Function
    if length(Y) === res_FD.N.tot
        Y_new = Y
    else
        Y_new = zeros(eltype(Y), length(res_FD.Y_cache))
        @inbounds @views Y_new[1:res_FD.N.diff] .= res_FD.Y_cache[1:res_FD.N.diff]
        @inbounds @views Y_new[res_FD.N.diff+1:end] .= Y
    end
    res_FD.f!(res, 0.0, Y_new, res_FD.YP_cache, res_FD.θ_tot)
end

struct jacobian_AD{T<:Function} <: AbstractJacobian
    f!::res_FD{T}
    sp::SparseMatrixCSC{Float64,Int64}
    jac_cache::ForwardColorJacCache
end
@inline function (jac::jacobian_AD{T})(t,Y,YP,γ::Float64,p::P,run) where {T<:Function,P<:AbstractParam}
    J = jac.sp
    forwarddiff_color_jacobian!(J, jac.f!, Y, jac.jac_cache)
    if size(J) === (p.N.tot-1,p.N.tot)
        @inbounds for i in 1:p.N.diff
            J[i,i] += -γ
        end
    end
    return nothing
end

struct jacobian_combined{
    T1<:Function,
    T2<:Union{SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false},SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}},
    T3<:Function,
    }
    sp::SparseMatrixCSC{Float64,Int64}
    base_func::T1
    J_base::T2
    scalar_func::T3
    J_scalar::SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}
    θ_tot::Vector{Float64}
    θ_keys::Vector{Symbol}
    L::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}
end
Base.getindex(J::jacobian_combined,i...) = getindex(J.sp,i...)
Base.axes(J::jacobian_combined,i...) = axes(J.sp,i...)

struct residual_combined{
    T1<:Function,
    T2<:Function,
    T3<:Function,
    }
    f_diff!::T1
    f_alg!::T2
    f_scalar!::T3
    ind_diff::UnitRange{Int64}
    ind_alg::UnitRange{Int64}
    θ_tot::Vector{Float64}
    θ_keys::Vector{Symbol}
end
struct Jac_and_res{T<:Sundials.IDAIntegrator}
    J_full::jacobian_combined
    R_full::residual_combined
    J_alg::jacobian_combined
    R_diff::residual_combined
    R_alg::residual_combined
    int::Vector{T}
end

Base.@kwdef mutable struct boundary_stop_conditions
    V_max::Float64 = -1.0
    V_min::Float64 = -1.0
    SOC_max::Float64 = -1.0
    SOC_min::Float64 = -1.0
    T_max::Float64 = -1.0
    c_s_n_max::Float64 = -1.0
    I_max::Float64 = NaN
    I_min::Float64 = NaN
    η_plating_min::Float64 = NaN
    c_e_min::Float64 = NaN
    dfilm_max::Float64 = NaN
    t_final_interp_frac::Float64 = +1.0
    V_prev::Float64 = -1.0
    SOC_prev::Float64 = -1.0
    T_prev::Float64 = -1.0
    c_s_n_prev::Float64 = -1.0
    I_prev::Float64 = -1.0
    η_plating_prev::Float64 = -1.0
    c_e_min_prev::Float64 = -1.0
    dfilm_prev::Float64 = -1.0
end

@inline function boundary_stop_conditions(V_max::Number, V_min::Number, SOC_max::Number, SOC_min::Number, T_max::Number, c_s_n_max::Number, I_max::Number, I_min::Number, η_plating_min::Number, c_e_min::Number, dfilm_max::Number)
    boundary_stop_conditions(
        Float64(V_max),
        Float64(V_min),
        Float64(SOC_max),
        Float64(SOC_min),
        Float64(T_max),
        Float64(c_s_n_max),
        Float64(I_max),
        Float64(I_min),
        Float64(η_plating_min),
        Float64(c_e_min),
        Float64(dfilm_max),
        +1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
end

mutable struct _funcs_numerical
    rxn_p::Function
    rxn_n::Function
    OCV_p::Function
    OCV_n::Function
    D_s_eff::Function
    D_eff::Function
    K_eff::Function
    thermodynamic_factor::Function
end
_funcs_numerical() = _funcs_numerical([emptyfunc for _ in fieldnames(_funcs_numerical)]...)

struct options_numerical{temp,solid_diff,Fickian,age}
    temperature::Bool
    solid_diffusion::Symbol
    Fickian_method::Symbol
    aging::Union{Bool,Symbol}
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
    thermodynamic_factor::Function
    jacobian::Symbol
end
options_numerical(temp,solid_diff,Fickian,age,x...) = 
    options_numerical{temp,solid_diff,Fickian,age}(temp,solid_diff,Fickian,age,x...)

const states_logic = model_states{
    Bool,
    Bool,
    Tuple
}

const indices_states = model_states{
    index_state,
    index_state,
    Nothing,
}

Base.@kwdef mutable struct options_model
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
    save_start::Bool = false
    var_keep::states_logic = model_states_logic()
end

struct save_start_info{T<:AbstractMethod}
    method::T
    SOC::Float64 # rounded to 3rd decimal place
    I::Float64 # rounded to 3rd decimal place
end

struct cache_run
    θ_tot::Vector{Float64}
    θ_keys::Vector{Symbol}
    cache_name::String
    state_labels::Vector{Symbol}
    vars::Tuple
    outputs_tot::Tuple
    save_start_dict::Dict{save_start_info,Vector{Float64}}
    Y0::Vector{Float64}
    YP0::Vector{Float64}
    res::Vector{Float64}
    Y_alg::Vector{Float64}
    id::Vector{Int64}
    constraints::Vector{Int64}
end

struct model_funcs{T1<:Function,T2<:Function,T3<:Function,T4<:AbstractJacobian,T5<:AbstractJacobian}
    initial_guess!::T1
    f_diff!::T2
    f_alg!::T3
    J_y!::T4
    J_y_alg!::T5
    Dict_constant::Dict{DataType,Jac_and_res}
    Dict_function::Dict{DataType,Dict{DataType,Jac_and_res}}
    Dict_residual::Dict{DataType,Jac_and_res}
end
model_funcs(x...) = model_funcs(x...,
    Dict{DataType,Jac_and_res}(),
    Dict{DataType,Dict{DataType,Jac_and_res}}(),
    Dict{DataType,Jac_and_res}()
    )
Base.empty!(f::model_funcs) = ([empty!(getproperty(f,field)) for (_type,field) in zip(fieldtypes(model_funcs),fieldnames(model_funcs)) if _type <: Dict];nothing)

struct param{T<:AbstractJacobian,temp,solid_diff,Fickian,age} <: AbstractParam{T,temp,solid_diff,Fickian,age}
    θ::Dict{Symbol,Float64}
    numerics::options_numerical{temp,solid_diff,Fickian,age}
    N::discretizations_per_section
    ind::indices_states
    opts::options_model
    bounds::boundary_stop_conditions
    cache::cache_run
    funcs::model_funcs{<:Function,<:Function,<:Function,T,<:AbstractJacobian}
end

const param_jac{jac}               = param{jac,<:Any,<:Any,<:Any,<:Any}
const param_temp{temp}             = param{<:AbstractJacobian,temp,<:Any,<:Any,<:Any}
const param_solid_diff{solid_diff} = param{<:AbstractJacobian,<:Any,solid_diff,<:Any,<:Any}
const param_Fickian{Fickian}       = param{<:AbstractJacobian,<:Any,<:Any,Fickian,<:Any}
const param_age{age}               = param{<:AbstractJacobian,<:Any,<:Any,<:Any,age}

const AbstractParamJac{jac}              = AbstractParam{jac,<:Any,<:Any,<:Any,<:Any}
const AbstractParamTemp{temp}            = AbstractParam{<:AbstractJacobian,temp,<:Any,<:Any,<:Any}
const AbstractParamSolidDiff{solid_diff} = AbstractParam{<:AbstractJacobian,<:Any,solid_diff,<:Any,<:Any}
const AbstractParamFickian{Fickian}      = AbstractParam{<:AbstractJacobian,<:Any,<:Any,Fickian,<:Any}
const AbstractParamAge{age}              = AbstractParam{<:AbstractJacobian,<:Any,<:Any,<:Any,age}

struct param_skeleton{temp,solid_diff,Fickian,age} <: AbstractParam{AbstractJacobian,temp,solid_diff,Fickian,age}
    θ::Dict{Symbol,Any}
    numerics::options_numerical{temp,solid_diff,Fickian,age}
    N::discretizations_per_section
    ind::indices_states
    opts::options_model
    bounds::boundary_stop_conditions
    cache::cache_run
end

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
    p::param
end

const model_output = model_states{
    Array{Float64,1},
    VectorOfArray{Float64,2,Array{Array{Float64,1},1}},
    Array{run_results,1},
}

Base.length(model::model_output) = length(model.results)
Base.isempty(model::model_output) = isempty(model.results)
function Base.getindex(model::T, i1::Int) where T<:model_output
    ind = (model.results[i1].run_index) .+ (1-model.results[1].run_index[1])
    T([fields === :results ? [model.results[i1]] : (x = getproperty(model, fields); length(x) > 1 ? x[ind] : x) for fields in fieldnames(T)]...)
end
function Base.getindex(model::T, i::UnitRange{Int64}) where T<:model_output
    ind = ((model.results[i[1]].run_index[1]):(model.results[i[end]].run_index[end])) .+ (1-model.results[1].run_index[1])
    T([fields === :results ? model.results[i] : (x = getproperty(model, fields); length(x) > 1 ? x[ind] : x) for fields in fieldnames(T)]...)
end
Base.lastindex(model::T) where T<:model_output = length(model)
Base.firstindex(::T) where T<:model_output = 1


## Modifying Base functions
@recipe function plot(model::model_output, x_name::Symbol=:V;linewidth=2,legend=false)
    x = getproperty(model, x_name)
    if x isa AbstractMatrix
        x = x'
    end
    
    if     x_name === :c_e
        ylabel = "Electrolyte Conc. (mol/m³)"
    elseif x_name === :c_s_avg
        ylabel = "Solid Conc. (mol/m³)"
    elseif x_name === :T
        ylabel = "Temperature (K)"
    elseif x_name === :film
        ylabel = "Li Plating Thickness (m)"
    elseif x_name === :Q
        ylabel = "Q"
    elseif x_name === :j
        ylabel = "Molar Ionic Flux (mol/m²⋅s)"
    elseif x_name === :j_s
        ylabel = "Side Reaction Flux (mol/m²⋅s)"
    elseif x_name === :Φ_e
        ylabel = "Electrolyte Potential (V)"
    elseif x_name === :Φ_s
        ylabel = "Solid-phase Potential (V)"
    elseif x_name === :I
        ylabel = "Current (C-rate)"
    elseif x_name === :V
        ylabel = "Voltage (V)"
    elseif x_name === :P
        ylabel = "Power (W)"
    elseif x_name === :SOC
        ylabel = "State-of-Charge (-)"
    else
        ylabel = "$x_name"
    end
    
    if length(model.t) ≠ length(x) error("$x_name is not in `outputs`") end
    
    time_unit, time_scale = time_units(model.t[end])[2:3]

    legend --> legend
    yguide --> ylabel
    xguide --> "Time ($(time_unit))"
    linewidth --> linewidth
    model.t./time_scale, x
end

function time_units(t::Number)
    if     t < 3600
        time_scale = 1.0
        time_unit = "s"
    elseif t < 3600*24
        time_scale = 3600.0
        time_unit = "hr"
    else
        time_scale = 3600.0*24
        time_unit = "days"
    end
    t /= time_scale
    return t, time_unit, time_scale
end

function C_rate_string(I::Number;digits::Int64=4)
    I_rat = rationalize(Float64(I))
    num = I_rat.num
    den = I_rat.den

    if den > 100 || (abs(num) > 10 && den > 10)
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
    
    # spacing
    sp = p.numerics.solid_diffusion === :Fickian ? "  " : ""

    # create the header for param
    if p isa param_skeleton
        header = "param_skeleton"
    else
        header = [x for x in replace(summary(p), "PETLION."=>"")]
        deleteat!(header, findall('{' .== header)[2]:length(header)-1)
        header = join(header)
    end
    
    str = string(
    "$header:\n",
    "  Cathode: $(p.numerics.cathode), $(p.numerics.rxn_p), & $(p.numerics.OCV_p)\n",
    "  Anode:   $(p.numerics.anode), $(p.numerics.rxn_n), & $(p.numerics.OCV_n)\n",
    "  System:  $(p.numerics.D_s_eff), $(p.numerics.rxn_rate), $(p.numerics.D_eff), $(p.numerics.K_eff), & $(p.numerics.thermodynamic_factor)\n",
    :model_methods ∈ fieldnames(typeof(p)) && !isempty(p.model_methods) ? 
    "  Methods: $(join(p.model_methods, ", "))\n" : "",
    "  --------\n",
    "  Temperature:     $(p.numerics.temperature)\n",
    "  Solid diffusion: $(p.numerics.solid_diffusion)",
    p.numerics.solid_diffusion === :Fickian ? 
    ", $(p.numerics.Fickian_method)\n" : "\n",
    "  Aging:           $(p.numerics.aging)\n",
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
    types = Symbol[]
    indices = UnitRange{Int64}[]

    for field in outputs_tot
        ind_var = getproperty(ind, field)
        if ind_var.var_type ∈ (:differential, :algebraic)
            push!(vars, field)
            push!(tot, ind_var.start)
            push!(types, ind_var.var_type)
            push!(indices, ind_var.start:ind_var.stop)
        end
    end
    vars    .= vars[sortperm(tot)]
    indices .= indices[sortperm(tot)]

    pad = maximum(length.(String.(vars)))+2

    str = [
        "indices_states:";
        ["  " * rpad("$(var): ", pad) * "$(length(index) > 1 ? index : index[1]), $(_type)" for (index,var,_type) in zip(indices,vars,types)]
    ]
    
    print(io, join(str, "\n"))
end
method_symbol(::Type{method_I}) = :I
method_symbol(::Type{method_V}) = :V
method_symbol(::Type{method_P}) = :P

method_name(::run_constant{method_I,<:Any};        shorthand::Bool=false) = shorthand ? "I"      : "current"
method_name(::run_constant{method_V,<:Any};        shorthand::Bool=false) = shorthand ? "V"      : "voltage"
method_name(::run_constant{method_P,<:Any};        shorthand::Bool=false) = shorthand ? "P"      : "power"
method_name(::run_function{method_I,<:Function};   shorthand::Bool=false) = shorthand ? "I func"   : "current function"
method_name(::run_function{method_V,<:Function};   shorthand::Bool=false) = shorthand ? "V func"   : "voltage function"
method_name(::run_function{method_P,<:Function};   shorthand::Bool=false) = shorthand ? "P func"   : "power function"

method_string(run::run_constant{method_I,<:Any};     kw...) = method_name(run;kw...) * " = $(C_rate_string(value(run);digits=2))"
method_string(run::run_constant{method_V,<:Any};     kw...) = method_name(run;kw...) * " = $(round(value(run);digits=2)) V"
method_string(run::run_constant{method_P,<:Any};     kw...) = method_name(run;kw...) * " = $(round(value(run);digits=2)) W"

function Base.show(io::IO, result::run_results{T}) where {T<:AbstractRun}
    run = result.run
    tspan = result.tspan

    fix(x, digits=2) = round(x, digits=digits)
    str = method_string(run)
    str *= " from ($(fix(tspan[1])) s, $(fix(tspan[2])) s)"

    print(io, str)
end

function Base.show(io::IO, run::T) where {T<:AbstractRun}
    str = "Run for $(method_string(run;shorthand=true))"
    if !isempty(run.info.exit_reason)
        str *= " with exit: $(run.info.exit_reason)"
    end
    
    print(io, str)
end
Base.show(io::IO, funcs::model_funcs) = println(io,"PETLION model functions")
function Base.show(io::IO, model::model_output)
    results = model.results
    p = results[1].p
    Y = @views @inbounds model.Y[end]
    function str_runs()
    
        str = length(results) === 1 ? "  Run: " : "  Runs:"
        str *= " "^4

        methods = method_name.([result.run for result in results];shorthand=true)

        counts = ones(Int64, length(methods))

        @inbounds for i in length(counts)-1:-1:1
            if methods[i] === methods[i+1]
                counts[i] += counts[i+1]
                deleteat!(methods, i+1)
                deleteat!(counts, i+1)
            end
        end

        max_methods = 7
        show_final = 2
        @inbounds for i in 1:min(length(methods),max_methods)
            methods[i] = (counts[i] === 1 ? methods[i] : "$(counts[i]) $(methods[i])")
        end
        if length(methods) > max_methods
            deleteat!(methods,max_methods-(show_final-1):length(methods)-show_final)
            insert!(methods, length(methods)-(show_final-1),"...")
        end
        str *= join(methods, " → ") * "\n"

        return str
    end

    t, time_unit = time_units(model.t[end])

    title = "PETLION model"
    if !isempty(model)
        str = @views @inbounds string(
                "$title\n",
                "  --------\n",
                str_runs(),
                "  Time:    $(round(t;                  digits = 2)) $time_unit\n",
                "  Voltage: $(round(calc_V(Y,p);        digits = 4)) V\n",
                "  Current: $(C_rate_string(calc_I(Y,p);digits = 4))\n",
                "  Power:   $(round(calc_P(Y,p);        digits = 4)) W\n",
                "  SOC:     $(round(model.SOC[end];     digits = 4))\n",
                !(p.numerics.temperature === false) ? 
                "  Temp.:   $(round(temperature_weighting(calc_T(Y,p),p)-273.15; digits = 4)) °C\n"
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
    Change matrix-vector multiplication in Symbolics to ignore 0s in the matrix.
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