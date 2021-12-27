const empty_vector_of_array = VectorOfArray(Array{Vector{Float64}}(Float64[]))
Base.@kwdef struct model_states{vec1D,vec2D,R1}
    """
    If you want to add anything to this struct, you must also check/modify set_vars!`.
    Otherwise, it may not work as intended
    """
    # Matrices (vectors in space and time)
    Y::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    YP::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    c_e::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    c_s_avg::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    T::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    film::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    Q::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    j::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    j_s::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    Φ_e::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    Φ_s::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? copy(empty_vector_of_array) : nothing
    # Vectors (vectors in time, not space)
    I::vec1D = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    t::vec1D = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    V::vec1D = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    P::vec1D = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    SOC::vec1D = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    SOH::vec1D = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    # Culmination of all the runs
    results::R1 = R1 === Array{run_results,1} ? run_results[] : nothing
end

@inline function model_states_logic(
    outputs = (),
    outputs_tot::Tuple = @inbounds @views (fieldnames(model_states)[1:end-1]...,)
    )
    """
    Creates the `keep` struct which determine what variables are calculated, kept for
    output, or discarded. See `set_vars!` for their use
    """

    if !isa(outputs, Tuple)
        outputs = (outputs,)
    end
    
    use_all = :all ∈ outputs

    x = @inbounds (use_all || field ∈ outputs for field in outputs_tot)
    
    return model_states{Bool,Bool,Tuple}(x..., outputs)
end
