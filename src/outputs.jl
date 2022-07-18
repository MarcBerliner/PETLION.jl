const empty_vector_of_array = VectorOfArray(Array{Vector{Float64}}(Float64[]))
assign_1D(vec1D) = ( vec1D == Array{Float64,1} ) ? Float64[] : nothing
assign_2D(vec2D) = ( vec2D == VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
Base.@kwdef struct solution_states{vec1D,vec2D,R1}
    """
    If you want to add anything to this struct, you must also check/modify set_vars!`.
    Otherwise, it may not work as intended
    """
    # Matrices (vectors in space and time)
    Y::vec2D = assign_2D(vec2D)
    YP::vec2D = assign_2D(vec2D)
    c_e::vec2D = assign_2D(vec2D)
    c_s_avg::vec2D = assign_2D(vec2D)
    T::vec2D = assign_2D(vec2D)
    film::vec2D = assign_2D(vec2D)
    δ::vec2D = assign_2D(vec2D)
    ϵ_s::vec2D = assign_2D(vec2D)
    Q::vec2D = assign_2D(vec2D)
    j::vec2D = assign_2D(vec2D)
    j_s::vec2D = assign_2D(vec2D)
    j_SEI::vec2D = assign_2D(vec2D)
    σ_h::vec2D = assign_2D(vec2D)
    Φ_e::vec2D = assign_2D(vec2D)
    Φ_s::vec2D = assign_2D(vec2D)
    # Vectors (vectors in time, not space)
    I::vec1D = assign_1D(vec1D)
    t::vec1D = assign_1D(vec1D)
    V::vec1D = assign_1D(vec1D)
    P::vec1D = assign_1D(vec1D)
    SOC::vec1D = assign_1D(vec1D)
    SOH::vec1D = assign_1D(vec1D)
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