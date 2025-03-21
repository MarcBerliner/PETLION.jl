const PETLION_VERSION = (Meta.parse.(split(String(Symbol(@PkgVersion.Version)),"."))...,)
const options = Dict{Symbol,Any}(
    :SAVE_SYMBOLIC_FUNCTIONS => true,
    :FILE_DIRECTORY => nothing,
    :FACTORIZATION_METHOD => :KLU, # :KLU or :LU
    :DIR_SAVED_MODELS => "saved_models",
)

# https://en.wikipedia.org/wiki/2019_redefinition_of_the_SI_base_units
const const_Faradays  = 96485.3321233
const const_Ideal_Gas = 8.31446261815324

abstract type AbstractJacobian <: Function end
abstract type AbstractModel{T<:AbstractJacobian,temp,solid_diff,Fickian,age} end
abstract type AbstractMethod end
abstract type AbstractRun{T<:AbstractMethod,input<:Any} end

struct method_I   <: AbstractMethod end
struct method_V   <: AbstractMethod end
struct method_P   <: AbstractMethod end
struct method_η_p <: AbstractMethod end
struct method_res <: AbstractMethod end

struct lithium_foil <: Function end

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
    iterations::Int64 = 1
end

struct run_constant{T<:AbstractMethod,in<:Union{Number,Symbol,Function}} <: AbstractRun{T,in}
    input::in
    value::Base.RefValue{Float64}
    method::T
    t0::Float64
    tf::Float64
    name::Symbol
    info::run_info
end
struct run_function{T<:AbstractMethod,func<:Function} <: AbstractRun{T,func}
    func::func
    value::Base.RefValue{Float64}
    method::T
    t0::Float64
    tf::Float64
    name::Symbol
    info::run_info
end
struct run_residual{T<:AbstractMethod,func<:Function} <: AbstractRun{T,func}
    func::func
    value::Base.RefValue{Float64}
    method::T
    t0::Float64
    tf::Float64
    name::Symbol
    info::run_info
end
@inline value(run::AbstractRun) = @inbounds run.value[]

Base.@kwdef mutable struct index_state <: AbstractUnitRange{Int64}
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

Base.@kwdef struct jacobian_symbolic{T<:Function} <: AbstractJacobian
    func::T = emptyfunc
    sp::SparseMatrixCSC{Float64,Int64} = spzeros(Float64,0,0)
end
(jac::jacobian_symbolic{<:Function})(x...) = jac.func(x...)

struct res_FD{T<:Function} <: Function
    f!::T
    Y_cache::Vector{Float64}
    YP_cache::Vector{Float64}
    Y_new::Vector{Float64}
    θ_tot::Vector{Float64}
    N::discretizations_per_section
end
@inline function (res_FD::res_FD{T1})(res::Vector{T2}, Y::Vector{T2}) where {T1<:Function,T2<:Number}
    if length(Y) == res_FD.N.tot
        Y_new = Y
    else
        Y_new = zeros(T2, res_FD.N.tot)
        @inbounds @views Y_new[1:res_FD.N.diff] .= res_FD.Y_cache[1:res_FD.N.diff]
        @inbounds @views Y_new[res_FD.N.diff+1:end] .= Y
    end
    res_FD.f!(res, 0.0, Y_new, res_FD.YP_cache, res_FD.θ_tot)
    return nothing
end

struct jacobian_AD{T<:Function,cache<:ForwardColorJacCache} <: AbstractJacobian
    f!::res_FD{T}
    sp::SparseMatrixCSC{Float64,Int64}
    jac_cache::cache
end
@inline function (jac::jacobian_AD{T1,<:ForwardColorJacCache})(t,Y::T2,YP,γ::Float64,p::AbstractModel,run) where {T1<:Function,T2<:SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}}
    # Jacobian for just the algebraic terms
    J = jac.sp
    forwarddiff_color_jacobian!(J, jac.f!, Y, jac.jac_cache)
    return nothing
end
@inline function (jac::jacobian_AD{T1,<:ForwardColorJacCache})(t,Y::T2,YP,γ::Float64,p::AbstractModel,run) where {T1<:Function,T2<:Vector{Float64}}
    # Jacobian for the differential and algebraic terms
    J = jac.sp
    forwarddiff_color_jacobian!(J, jac.f!, Y, jac.jac_cache)
    @inbounds for i in 1:p.N.diff
        J[i,i] += -γ
    end
    return nothing
end

struct jacobian_combined{
    T1<:Function,
    T2<:Union{SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false},SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}},
    T3<:Function,
    T4<:LinearAlgebra.Factorization{Float64},
    }
    sp::SparseMatrixCSC{Float64,Int64}
    base_func::T1
    J_base::T2
    scalar_func::T3
    J_scalar::SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}
    θ_tot::Vector{Float64}
    θ_keys::Vector{Symbol}
    factor::T4
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

Base.@kwdef mutable struct boundary_stop_prev_values
    t_final_interp_frac::Float64 = +1.0
    V::Float64 = -1.0
    SOC::Float64 = -1.0
    T::Float64 = -1.0
    c_s_n::Float64 = -1.0
    I::Float64 = -1.0
    η_plating::Float64 = -1.0
    c_e_min::Float64 = -1.0
    dfilm::Float64 = -1.0
end

remove_module_name(x::String) = replace(x, "$(@__MODULE__)."=>"")
function create_immutable_version(structure::DataType; str_replacements=(""=>"",), conv_replacements=(""=>"",))
    """
    Immutable versions of struct can be beneficial for performance. This function
    creates an immutable version of mutable structures with "_immutable" appended to the end.
    `str_replacements` uses the `replace` function to substitute string values in the evaluated
    struct.
    """

    name = String(Symbol(structure))
    fields = String.(fieldnames(structure))
    types = String.(Symbol.(fieldtypes(structure)))
    
    args = join([field * "::" * type for (field,type) in zip(fields,types)], "\n")

    super = supertype(structure)

    # string to create new struct
    str_immutable = "struct $(name)_immutable <: $super
    $(args)
    end"

    if str_replacements isa Pair{String, String}
        str_replacements = (str_replacements,)
    end
    if conv_replacements isa Pair{String, String}
        conv_replacements = (conv_replacements,)
    end

    
    # string to convert the mutable struct to the immutable struct  
    conversion = "$(name)_immutable(x::$(name)) = $(name)_immutable($(join(map(field->"x.$field", fields), ",")))"
    
    for replacement in str_replacements
        str_immutable = replace(str_immutable, replacement)
    end
    for replacement in conv_replacements
        conversion = replace(conversion, replacement)
    end

    # The names cannot be prepended by "PETLION.", otherwise it creates errors
    str_immutable = remove_module_name(str_immutable)
    conversion = remove_module_name(conversion)

    eval.(Meta.parse.((str_immutable,conversion)))

    return nothing
end
create_immutable_version(index_state)

abstract type AbstractStopConditions end
Base.@kwdef mutable struct boundary_stop_conditions <: AbstractStopConditions
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
    prev::boundary_stop_prev_values = boundary_stop_prev_values()
end
create_immutable_version(boundary_stop_conditions; conv_replacements=", x.prev" => ", x.$(fieldtypes(boundary_stop_conditions)[end])()")

const states_logic = solution_states{
    Bool,
    Bool,
    <:Tuple
}

const indices_states = solution_states{
    index_state_immutable,
    index_state_immutable,
    Tuple,
}

abstract type AbstractOptionsModel end
Base.@kwdef mutable struct options_simulation <: AbstractOptionsModel
    outputs = (:t, :V)
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
    tstops::Vector{<:Number} = Float64[]
    tdiscon::Vector{<:Number} = Float64[]
    interp_bc::Symbol = :interpolate
    save_start::Bool = false
    var_keep::states_logic = solution_states_logic(outputs)[1]
    stop_function::Function = (x...) -> nothing
    calc_integrator::Bool = false
end

create_immutable_version(options_simulation; str_replacements=(
    "_immutable" =>"_immutable{T<:Function}",
    "stop_function::Function" => "stop_function::T",
    "outputs::Any" => "outputs::Tuple",
    ))


struct save_start_info{T<:AbstractMethod}
    method::T
    SOC::Float64 # rounded to 3rd decimal place
    I::Float64 # rounded to 3rd decimal place
end

struct cache_run
    θ_tot::Vector{Float64}
    θ_keys::Vector{Symbol}
    state_labels::Vector{Symbol}
    vars::Tuple
    save_start_dict::Dict{save_start_info,Vector{Float64}}
    Y0::Vector{Float64}
    Y_full::Vector{Float64}
    YP0::Vector{Float64}
    res::Vector{Float64}
    Y_alg::Vector{Float64}
    outputs_possible::Tuple
    id::Vector{Int64}
end

Base.@kwdef struct model_funcs{T1<:Function,T2<:Function,T3<:Function,T4<:AbstractJacobian,T5<:AbstractJacobian}
    initial_guess!::T1 = emptyfunc
    f_diff!::T2 = emptyfunc
    f_alg!::T3 = emptyfunc
    J_y!::T4 = jacobian_symbolic()
    J_y_alg!::T5 = jacobian_symbolic()
    Dict_constant::Dict{DataType,Jac_and_res} = Dict{DataType,Jac_and_res}()
    Dict_function::Dict{DataType,Dict{DataType,Jac_and_res}} = Dict{DataType,Dict{DataType,Jac_and_res}}()
    Dict_residual::Dict{DataType,Jac_and_res} = Dict{DataType,Jac_and_res}()
end
model_funcs(initial_guess!,f_diff!,f_alg!,J_y!,J_y_alg!) = model_funcs(
    initial_guess!,
    f_diff!,
    f_alg!,
    J_y!,
    J_y_alg!,
    Dict{DataType,Jac_and_res}(),
    Dict{DataType,Dict{DataType,Jac_and_res}}(),
    Dict{DataType,Jac_and_res}()
    )

struct model{T<:AbstractJacobian,temp,solid_diff,Fickian,age} <: AbstractModel{T,temp,solid_diff,Fickian,age}
    θ::OrderedDict{Symbol,Float64}
    numerics::options_numerical{temp,solid_diff,Fickian,age}
    N::discretizations_per_section
    ind::indices_states
    opts::options_simulation
    bounds::boundary_stop_conditions
    cache::cache_run
    funcs::model_funcs{<:Function,<:Function,<:Function,T,<:AbstractJacobian}
end

const model_jac{jac}               = model{jac,<:Any,<:Any,<:Any,<:Any}
const model_temp{temp}             = model{<:AbstractJacobian,temp,<:Any,<:Any,<:Any}
const model_solid_diff{solid_diff} = model{<:AbstractJacobian,<:Any,solid_diff,<:Any,<:Any}
const model_Fickian{Fickian}       = model{<:AbstractJacobian,<:Any,<:Any,Fickian,<:Any}
const model_age{age}               = model{<:AbstractJacobian,<:Any,<:Any,<:Any,age}

const AbstractModelJac{jac}              = AbstractModel{jac,<:Any,<:Any,<:Any,<:Any}
const AbstractModelTemp{temp}            = AbstractModel{<:AbstractJacobian,temp,<:Any,<:Any,<:Any}
const AbstractModelSolidDiff{solid_diff} = AbstractModel{<:AbstractJacobian,<:Any,solid_diff,<:Any,<:Any}
const AbstractModelFickian{Fickian}      = AbstractModel{<:AbstractJacobian,<:Any,<:Any,Fickian,<:Any}
const AbstractModelAge{age}              = AbstractModel{<:AbstractJacobian,<:Any,<:Any,<:Any,age}

struct model_skeleton{temp,solid_diff,Fickian,age} <: AbstractModel{AbstractJacobian,temp,solid_diff,Fickian,age}
    θ::OrderedDict{Symbol,Any}
    numerics::options_numerical{temp,solid_diff,Fickian,age}
    N::discretizations_per_section
    ind::indices_states
    opts::options_simulation
    bounds::boundary_stop_conditions
    cache::cache_run
end

struct run_results{T<:AbstractRun}
    run::T
    tspan::Tuple{Float64,Float64}
    info::run_info
    run_index::UnitRange{Int64}
    int::Sundials.IDAIntegrator
    opts::options_simulation_immutable
    bounds::boundary_stop_conditions_immutable
    N::discretizations_per_section
    numerics::options_numerical
    p::model
end

const solution = solution_states{
    Array{Float64,1},
    VectorOfArray{Float64,2,Array{Array{Float64,1},1}},
    Array{run_results,1},
}

Base.length(sol::solution) = length(sol.results)
Base.isempty(sol::solution) = isempty(sol.results)
function Base.getindex(sol::T, i1::Int) where T<:solution
    ind = (sol.results[i1].run_index) .+ (1-sol.results[1].run_index[1])
    T([fields == :results ? [sol.results[i1]] : (x = getproperty(sol, fields); length(x) > 1 ? x[ind] : x) for fields in fieldnames(T)]...)
end
function Base.getindex(sol::T, i::UnitRange{Int64}) where T<:solution
    ind = ((sol.results[i[1]].run_index[1]):(sol.results[i[end]].run_index[end])) .+ (1-sol.results[1].run_index[1])
    T([fields == :results ? sol.results[i] : (x = getproperty(sol, fields); length(x) > 1 ? x[ind] : x) for fields in fieldnames(T)]...)
end
Base.lastindex(sol::T) where T<:solution = length(sol)
Base.firstindex(::T) where T<:solution = 1

Base.empty!(f::model_funcs) = ([empty!(getproperty(f,field)) for (_type,field) in zip(fieldtypes(model_funcs),fieldnames(model_funcs)) if _type <: Dict];nothing)
Base.empty!(p::model) = empty!(p.funcs)

const STATE_NAMES = Dict{Symbol,String}(
    :c_e => "Electrolyte Conc. (mol/m³)",
    :c_s_avg => "Solid Conc. (mol/m³)",
    :T => "Temperature (K)",
    :film => "SEI Film Thickness (m)",
    :Q => "Q",
    :j => "Molar Ionic Flux (mol/m²⋅s)",
    :j_s => "Side Reaction Flux (mol/m²⋅s)",
    :Φ_e => "Electrolyte Potential (V)",
    :Φ_s => "Solid-phase Potential (V)",
    :t => "Time (s)",
    :I => "Current (C-rate)",
    :V => "Voltage (V)",
    :P => "Power (W)",
    :SOC => "State of Charge (-)",
    :SOH => "State of Health (-)",
)

## Modifying Base functions
@recipe function plot(sol::solution, x_name::Union{Symbol,String}=:V;linewidth=2,legend=false)
    x_name = Symbol(x_name)
    if x_name ∉ fieldnames(solution)
        error("$x_name is not a valid state")
    elseif !getproperty(sol.results[end].opts.var_keep, x_name)
        error("$x_name is not in `outputs`")
    end

    x = getproperty(sol, x_name)
    if x isa VectorOfArray
        x = Matrix(x')
    end
    
    if haskey(STATE_NAMES, x_name)
        ylabel = STATE_NAMES[x_name]
    else
        ylabel = "$x_name"
    end
    
    legend --> legend
    yguide --> ylabel
    xguide --> "Time (s)"
    linewidth --> linewidth
    sol.t, x
end

function time_units(t::Number; digits::T=nothing) where T
    # round the number to the nearest digits
    rounding = t -> T <: Number ? round(t; digits=digits) : t
    
    if     rounding(t) < 3600
        time_scale = 1.0
        time_unit = "s"
    elseif rounding(t) < 3600*24
        time_scale = 3600.0
        time_unit = "hr"
    else
        time_scale = 3600.0*24
        time_unit = "days"
    end
    t /= time_scale
    t = rounding(t)
    return t, time_unit, time_scale
end

function C_rate_string(I::Number; show_sign::Bool=true)
    I_rat = rationalize(float(I))
    num = abs(I_rat.num)
    den = I_rat.den

    if den > 100 || (num > 10 && den > 10)
        str = "$(show_sign ? I : abs(I))C"
        return str
    end
    
    str = ""
    if show_sign && I < 0
        str *=  "-"
    end
    
    if isone(num) && !isone(den)
        str *= "C"
    else
        str *= "$(num)C"
    end
    
    if !isone(den)
        str *= "/$(den)"
    end
    
    return str
end

function Base.show(io::IO, p::AbstractModel)
    function show_bounds(title, min, max, units="")
        if isnan(min) && isnan(max)
            return ""
        end
        
        pad = 18
        str = " $title "
        if isnan(max)
            str = rpad(str * "min:",pad)*"$min$units\n"
        elseif isnan(min)
            str = rpad(str * "max:",pad)*"$max$units\n"
        else
            str = rpad(str * "bounds:",pad)*"[$min$units, $max$units]\n"
        end
        
        return str
        
    end

    # create the header for model
    if p isa model_skeleton
        header = "model"
    else
        header = [x for x in remove_module_name(summary(p))]
        deleteat!(header, findall('{' .== header)[2]:length(header)-1)
        header = join(header)
    end
    header = "$(@__MODULE__) $header"

    lim_electrode, Q_min = try
        limiting_electrode(p)
    catch
        "Unknown", NaN
    end

    
    str = string(
    "$header:\n",
    " Cathode: $(p.numerics.cathode), $(p.numerics.rxn_p), & $(p.numerics.OCV_p)\n",
    " Anode:   $(p.numerics.anode), $(p.numerics.rxn_n), & $(p.numerics.OCV_n)\n",
    " System:  $(p.numerics.D_s_eff), $(p.numerics.rxn_rate), $(p.numerics.D_eff), $(p.numerics.K_eff), & $(p.numerics.thermodynamic_factor)\n",
    " ---Options---\n",
    " Temperature:     $(p.numerics.temperature)\n",
    " Solid diffusion: $(p.numerics.solid_diffusion)",
    p.numerics.solid_diffusion == :Fickian ? 
    ", $(p.numerics.Fickian_method)\n" : "\n",
    " Aging:           $(p.numerics.aging)\n",
    " Total capacity:  $(round(Q_min;digits=2)) Ah/m², $(lim_electrode)-limited\n",
    show_bounds("Voltage", p.bounds.V_min, p.bounds.V_max, " V"),
    show_bounds("SOC", p.bounds.SOC_min, p.bounds.SOC_max),
    show_bounds("Current", p.bounds.I_min, p.bounds.I_max, "C"),
    p.numerics.temperature ?
    show_bounds("Temperature", NaN, p.bounds.T_max-273.15, " °C") : "",
    show_bounds("Anode sat.", NaN, p.bounds.c_s_n_max),
    " ---Node Points---\n",
    p.numerics.temperature ?
    " Pos. tab:       $(p.N.a)\n" : "",
    " Pos. electrode: $(p.N.p)\n",
    p.numerics.solid_diffusion == :Fickian ?
    " Pos. particles: $(p.N.r_p)\n" : "",
    " Separator:      $(p.N.s)\n",
    " Neg. electrode: $(p.N.n)\n",
    p.numerics.solid_diffusion == :Fickian ?
    " Neg. particles: $(p.N.r_n)\n" : "",
    p.numerics.temperature ?
    " Neg. tab:       $(p.N.z)\n" : "",
    )
    
    print(io, str[1:end-1]) # removes the final \n
end

function Base.show(io::IO, ind::indices_states)
    
    names = [fieldnames(indices_states)...]

    X = [getfield(ind, name) for name in names]

    isnt_used(x::index_state_immutable) = x.var_type == :NA
    isnt_used(::Any) = true

    inds_inactive = isnt_used.(X)

    deleteat!(X, inds_inactive)
    deleteat!(names, inds_inactive)

    ind_sort = sortperm(X)

    X .= X[ind_sort]
    names .= names[ind_sort]

    len = maximum(length.(String.(names)))

    index_range(x) = x.start == x.stop ? "$(x.stop)" : "$(x.start):$(x.stop)"

    str = "indices_states:\n" * join([rpad(" $(names[i]): ", len+3) * index_range(X[i]) * ", $(X[i].var_type)" for i in 1:length(names)], "\n")
    
    print(io, str)
end
function Base.show(io::IO, bounds::T) where T<:AbstractStopConditions
    fields = fieldnames(T)[findall(fieldtypes(T) .<: Number)]
    vals = [getproperty(bounds,field) for field in fields]
    ind_remove = findall(.!isnan.(vals))
    fields = fields[ind_remove]
    vals = vals[ind_remove]

    pad = maximum(length.(String.(fields)))+2

    str = [
        "$T:";
        [" " * rpad("$(field): ", pad) * "$(val)" for (field,val) in zip(fields,vals)]
    ]
    
    print(io, join(str, "\n"))
end
function Base.show(io::IO, opts::T) where T<:Union{options_numerical,AbstractOptionsModel,run_info}
    fields = fieldnames(T)
    vals = [getproperty(opts,field) for field in fields]

    pad = maximum(length.(String.(fields)))+2

    str = [
        "$T:";
        [" " * rpad("$(field): ", pad) * "$(val)" for (field,val) in zip(fields,vals)]
    ]
    
    print(io, join(str, "\n"))
end

method_name(::run_constant{method_I,<:Any};        shorthand::Bool=false) = shorthand ? "I"      : "current"
method_name(::run_constant{method_V,<:Any};        shorthand::Bool=false) = shorthand ? "V"      : "voltage"
method_name(::run_constant{method_P,<:Any};        shorthand::Bool=false) = shorthand ? "P"      : "power"
method_name(::run_function{method_I,<:Function};   shorthand::Bool=false) = shorthand ? "I func"   : "current function"
method_name(::run_function{method_V,<:Function};   shorthand::Bool=false) = shorthand ? "V func"   : "voltage function"
method_name(::run_function{method_P,<:Function};   shorthand::Bool=false) = shorthand ? "P func"   : "power function"
method_name(::run_constant{method_η_p,<:Any};      shorthand::Bool=false) = shorthand ? "η_p"    : "plating overpotential"
method_name(::run_function{method_η_p,<:Function}; shorthand::Bool=false) = shorthand ? "η_p func" : "plating overpotential function"
method_name(run::run_residual;                     shorthand::Bool=false) = run.name == "res" ? (shorthand ? "I res"  : "current residual") : run.name

method_string(run::run_constant{method_I,<:Any};     kw...) = method_name(run;kw...) * " = $(C_rate_string(round(value(run);digits=2)))"
method_string(run::run_constant{method_V,<:Any};     kw...) = method_name(run;kw...) * " = $(round(value(run);digits=2)) V"
method_string(run::run_constant{method_P,<:Any};     kw...) = method_name(run;kw...) * " = $(round(value(run);digits=2)) W"

method_string(run::Union{run_residual,run_function}; kw...) = method_name(run;kw...)

function Base.show(io::IO,  ::MIME"text/plain", results::T) where T<:AbstractArray{run_results}
    str = remove_module_name(summary(results)) * ":\n"
    result_strs = [" $i: " * result_str(result) for (i,result) in enumerate(results)]
    str *= join(result_strs, "\n")
    
    print(io, str)
end
function result_str(result::run_results)
    run = result.run
    tspan = result.tspan
    
    fix(x, digits=2) = round(x, digits=digits)
    str = string(method_string(run))
    str *= " from ($(fix(tspan[1])) s, $(fix(tspan[2])) s)"

    return str
end
remove_string_types(str::String) = str[1:findfirst('{',str)-1]
function Base.show(io::IO, result::T) where T<:run_results
    str = remove_string_types("$T") * ": " * result_str(result) * ":"
    str *= "\n Fields: " * join(":" .* string.(fieldnames(T)), ", ")

    print(io, str)
end

function Base.show(io::IO, run::T) where {T<:AbstractRun}
    str = remove_string_types("$T") * ": " * "Run for $(method_string(run;shorthand=true))"
    if !isempty(run.info.exit_reason)
        str *= " with exit: $(run.info.exit_reason)"
    end
    str *= "\n Fields: " * join(":" .* string.(fieldnames(T)), ", ")
    
    print(io, str)
end

(funcs::model_funcs)(sol::solution) = (@assert !isempty(sol); (@inbounds funcs(sol.results[end].run)))
Base.show(io::IO, ::model_funcs) = println(io,"$(@__MODULE__) model functions")
function Base.show(io::IO, sol::solution)
    results = sol.results
    if !isempty(sol)
        p = results[1].p
        Y = @inbounds sol.Y[end]
        t, time_unit = time_units(sol.t[end]; digits=2)
    end
    function str_runs()
    
        str = " Run" * (length(results) == 1 ? ": " : "s:") * " "^4

        methods = method_name.([result.run for result in results];shorthand=true)

        counts = ones(Int64, length(methods))

        @inbounds for i in length(counts)-1:-1:1
            if methods[i] == methods[i+1]
                counts[i] += counts[i+1]
                deleteat!(methods, i+1)
                deleteat!(counts, i+1)
            end
        end

        max_methods = 6
        show_final = 3
        methods .= @inbounds [(counts[i] == 1 ? methods[i] : "$(counts[i]) $(methods[i])") for i in 1:length(methods)]
        
        if length(methods) > (max_methods+1)
            first_range = 1:(max_methods-show_final)
            final_range = (length(counts) - (show_final-1)):length(counts)
            count_not_showed = sum(counts) - sum(counts[[first_range;final_range]])

            methods = [
                methods[first_range];
                "…$(count_not_showed)…";
                methods[final_range];
            ]
        end
        str *= join(methods, " → ") * "\n"

        return str
    end

    title = "$(@__MODULE__) simulation"
    if !isempty(sol)
        str = @inbounds string(
                "$title\n",
                " --------\n",
                str_runs(),
                " Time:    $(t) $(time_unit)\n",
                " Current: $(C_rate_string(round(calc_I(Y,p); digits = 4)))\n",
                " Voltage: $(round(calc_V(Y,p);               digits = 4) ) V\n",
                " Power:   $(round(calc_P(Y,p);               digits = 2) ) W/m²\n",
                " SOC:     $(round(sol.SOC[end];              digits = 4) )\n",
                !(p.numerics.aging == false) ? 
                " SOH:     $(round(sol.SOH[end];              digits = 4) )\n"
                : "",
                p.numerics.temperature ? 
                # " Temp.:   $(round(calc_T_avg(Y,p)-273.15; digits = 4)) °C\n"
                " Temp.:   $(round(temperature_weighting(calc_T(Y,p),p)-273.15; digits = 4)) °C\n"
                : "",
                " Exit:    $(sol.results[end].info.exit_reason)",
            )
    else
        str = "$title: empty"
    end
    
    print(io, str)
end

Base.show(io::IO, ind::states_logic) = print(io, remove_module_name("states_logic using $(ind.results)"))

Base.deleteat!(a::VectorOfArray, i::Integer) = (Base._deleteat!(a.u, i, 1); a.u)
