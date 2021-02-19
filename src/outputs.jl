@with_kw struct model_states{vec1D,vec2D,R1}
    """
    If you want to add anything to this struct, you must also
    check/modify set_vars!`. Otherwise it may not work as intended
    """
    # Matrices (vectors in space and time)
    Y::vec2D       = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    YP::vec2D      = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    c_e::vec2D     = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    c_s_avg::vec2D = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    T::vec2D       = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    film::vec2D    = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    Q::vec2D       = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    j::vec2D       = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    j_s::vec2D     = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    Φ_e::vec2D     = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    Φ_s::vec2D     = ( vec2D === VectorOfArray{Float64,2,Array{Array{Float64,1},1}} ) ? VectorOfArray(Array{Vector{Float64}}([])) : nothing
    # Vectors (vectors in time, not space)
    I::vec1D   = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    t::vec1D   = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    V::vec1D   = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    P::vec1D   = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    SOC::vec1D = ( vec1D === Array{Float64,1} ) ? Float64[] : nothing
    # Culmination of all the runs
    results::R1 = R1 === Array{run_results,1} ? run_results[] : nothing
end

@inline function model_states_logic(
    outputs=(),
    outputs_tot::Tuple = (fieldnames(model_states)[1:end-1]...,) # the final entry is runs which is not an output
    )
    """
    Creates the keep` structs which determine what variables
    are calculated, kept for output, or discarded. See `set_vars!` for
    their use.
    """

    init_all = outputs === :all || :all ∈ outputs

    fields = outputs_tot

    x = Vector{Bool}(undef, length(fields))
    i = 1
    for field in fields
        @inbounds @views x[i] = init_all || field ∈ outputs
        i += 1
    end

    return model_states{Bool,Bool,Nothing}(x..., nothing)
end