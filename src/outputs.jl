const empty_vector_of_array = VectorOfArray(Array{Vector{Float64}}(Float64[]))
Base.@kwdef struct solution_states{vec1D,vec2D,R1}
    """
    If you want to add anything to this struct, you must also check/modify set_vars!`.
    Otherwise, it may not work as intended
    """
    # Matrices (vectors in space and time)
    Y::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    YP::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    c_e::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    c_s_avg::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    T::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    film::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    Q::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    j::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    j_s::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    Φ_e::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    Φ_s::vec2D = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    # Vectors (vectors in time, not space)
    I::vec1D = ( vec1D == Array{Float64,1} ) ? Float64[] : nothing
    t::vec1D = ( vec1D == Array{Float64,1} ) ? Float64[] : nothing
    V::vec1D = ( vec1D == Array{Float64,1} ) ? Float64[] : nothing
    P::vec1D = ( vec1D == Array{Float64,1} ) ? Float64[] : nothing
    SOC::vec1D = ( vec1D == Array{Float64,1} ) ? Float64[] : nothing
    SOH::vec1D = ( vec1D == Array{Float64,1} ) ? Float64[] : nothing
    # Culmination of all the runs
    results::R1 = R1 == Array{run_results,1} ? run_results[] : nothing
end

@inline solution_states_logic(output::Symbol) = solution_states_logic((output,))
eval(quote
@inline function solution_states_logic(outputs::T=()) where T<:Tuple
    outputs_tot = $(fieldnames(solution_states)[1:end-1])
    
    use_all = :all ∈ outputs

    x = @inbounds (use_all || field ∈ outputs for field in outputs_tot)
    
    return solution_states{Bool,Bool,T}(x..., outputs), use_all ? outputs : outputs
end
end)