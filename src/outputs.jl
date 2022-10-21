function emptyfunc end
Base.@kwdef mutable struct _funcs_numerical
    rxn_p::Function = emptyfunc
    rxn_n::Function = emptyfunc
    OCV_p::Function = emptyfunc
    OCV_n::Function = emptyfunc
    D_s_eff::Function = emptyfunc
    D_eff::Function = emptyfunc
    K_eff::Function = emptyfunc
    thermodynamic_factor::Function = emptyfunc
end

Base.@kwdef struct options_numerical{temp,solid_diff,Fickian,age}
    temperature::Bool = false
    solid_diffusion::Symbol = :NA
    Fickian_method::Symbol = :NA
    aging::Union{Bool,Symbol} = false
    cathode::Function = emptyfunc
    anode::Function = emptyfunc
    rxn_p::Function = emptyfunc
    rxn_n::Function = emptyfunc
    OCV_p::Function = emptyfunc
    OCV_n::Function = emptyfunc
    D_s_eff::Function = emptyfunc
    rxn_rate::Function = emptyfunc
    D_eff::Function = emptyfunc
    K_eff::Function = emptyfunc
    thermodynamic_factor::Function = emptyfunc
    jacobian::Symbol = :NA
end
options_numerical(temp,solid_diff,Fickian,age,x...) = 
    options_numerical{temp,solid_diff,Fickian,age}(temp,solid_diff,Fickian,age,x...)

Base.@kwdef mutable struct variable_def
    var_type = :NA
    sections = ()
    is_active = false
end

function model_states_and_outputs(numerics::options_numerical=options_numerical();
    remove_inactive::Bool=false,
    )
    """
    Defines the states which are active, the state type,
    and what sections the state is located in
    """

    states_old, outputs = model_variables(numerics)

    # Reorder the states to force :I to the end
    states = similar(states_old)
    for key in keys(states_old)
        if key ≠ :I
            states[key] = states_old[key]
        end
    end
    states[:I] = states_old[:I]
    
    for dict in (states,outputs), key in keys(dict)
        # Put all single symbols into a tuple
        if dict[key].sections isa Symbol
            dict[key].sections = (dict[key].sections,)
        end
        
        if remove_inactive && !dict[key].is_active
            delete!(dict, key)
        end
    end

    return states, outputs
end

const empty_vector_of_array = VectorOfArray(Array{Vector{Float64}}(Float64[]))
assign_1D(vec1D) = ( vec1D == Array{Float64,1} ) ? Float64[] : nothing
assign_2D(vec2D) = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing

const states_total, outputs_total = model_states_and_outputs()
begin
    var_types = map(x->states_total[x].var_type, Symbol.(keys(states_total)))
    type_string(x) = isempty(x.sections) ? "vec1D = assign_1D(vec1D)" : "vec2D = assign_2D(vec2D)"
    solution_states_vec = String[]
    for var_type in (:differential, :algebraic)
        inds = findall(isequal.(var_types, var_type))
        for state in Symbol.(keys(states_total))[inds]
            str = "$state::" * type_string(states_total[state])
            push!(solution_states_vec, str)
        end
    end
    for state in keys(outputs_total)
        str = "$state::" * type_string(outputs_total[state])
        push!(solution_states_vec, str)
    end
    return solution_states_vec
end

eval(quote
Base.@kwdef struct solution_states{vec1D,vec2D,R1}
    Y::vec2D = assign_2D(vec2D)
    YP::vec2D = assign_2D(vec2D)
    t::vec1D = assign_1D(vec1D)
    $(Meta.parse.(solution_states_vec)...)
    # Culmination of all the runs
    results::R1 = R1 == Array{run_results,1} ? run_results[] : nothing
end
end)

@inline solution_states_logic(output::Symbol,x...) = solution_states_logic((output,),x...)
eval(quote
@inline function solution_states_logic(outputs::T=()) where T<:Tuple
    outputs_tot = $(fieldnames(solution_states)[1:end-1])
    
    use_all = :all ∈ outputs

    x = @inbounds (use_all || field ∈ outputs for field in outputs_tot)
    
    return solution_states{Bool,Bool,T}(x..., outputs), outputs
end
end)

eval(quote
@inline function solution_states_logic(outputs::T,outputs_possible::NTuple{N,Symbol}) where {T<:Tuple,N}
    outputs_tot = $(fieldnames(solution_states)[1:end-1])
    
    use_all = :all ∈ outputs

    x = @inbounds (field ∈ outputs_possible && (use_all || field ∈ outputs) for field in outputs_tot)
    
    return solution_states{Bool,Bool,T}(x..., outputs), outputs
end
end)
