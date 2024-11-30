function model_variables(numerics)
    """
    Defines the states which are active, the state type,
    and what sections the state is located in.
    A `calc_state(Y, p)` function must also be defined for each output, e.g. 
        `@inline calc_V(Y::Vector{<:Number}, p::AbstractModel) = @inbounds Y[p.ind.Φ_s[1]] - Y[p.ind.Φ_s[end]]`
    """

    # States are directly calculated and solve for in the model
    states = OrderedDict{Symbol,variable_def}()

    states[:c_e] = variable_def(
        var_type = :differential,
        sections = (:p, :s, :n),
        is_active = true,
    )
    states[:c_s_avg] = variable_def(
        var_type = :differential,
        sections = numerics.solid_diffusion == :Fickian ? (:particle_p, :particle_n) : (:p, :n),
        is_active = true,
    )
    states[:j] = variable_def(
        var_type = :algebraic,
        sections = (:p, :n),
        is_active = true,
    )
    states[:Φ_e] = variable_def(
        var_type = :algebraic,
        sections = (:p, :s, :n),
        is_active = true,
    )
    states[:Φ_s] = variable_def(
        var_type = :algebraic,
        sections = (:p, :n),
        is_active = true,
    )
    states[:I] = variable_def(
        var_type = :algebraic,
        sections = (),
        is_active = true,
    )

    states[:T] = variable_def(
        var_type = :differential,
        sections = (:a, :p, :s, :n, :z),
        is_active = numerics.temperature,
    )    
    states[:j_s] = variable_def(
        var_type = :algebraic,
        sections = :n,
        is_active = numerics.aging == :SEI,
    )
    states[:film] = variable_def(
        var_type = :differential,
        sections = :n,
        is_active = numerics.aging == :SEI,
    )
    states[:SOH] = variable_def(
        var_type = :differential,
        sections = (),
        is_active = numerics.aging == :SEI,
    )
    states[:Q] = variable_def(
        var_type = :differential,
        sections = (:p, :n),
        is_active = numerics.solid_diffusion == :polynomial,
    )

    # Outputs are not calculated in the model, but are calculated after the model is solved.
    # `calc_state` MUST be defined for each output (see the example above)
    outputs = OrderedDict{Symbol,variable_def}()

    outputs[:SOC] = variable_def(var_type = :output, sections = (), is_active = true) # uses function calc_SOC(Y, p)
    outputs[:V]   = variable_def(var_type = :output, sections = (), is_active = true) # uses function calc_V(Y, p)
    outputs[:P]   = variable_def(var_type = :output, sections = (), is_active = true) # uses function calc_P(Y, p)
    
    return states, outputs
end

function initial_guess(p)
    """
    Get the initial guess in the DAE initialization.
    This function is made symbolic by Symbolics and 
    optionally saved as `initial_guess.jl`
    """

    states = retrieve_states(p)
    
    build_T!(states, p)

    SOC = p.opts.SOC

    states[:c_s_avg].p .= p.θ[:c_max_p] * (SOC*(p.θ[:θ_max_p] - p.θ[:θ_min_p]) + p.θ[:θ_min_p])
    states[:c_s_avg].n .= p.θ[:c_max_n] * (SOC*(p.θ[:θ_max_n] - p.θ[:θ_min_n]) + p.θ[:θ_min_n])
    
    # differential
    states[:c_e] .= p.θ[:c_e₀]
        
    states[:T] .= p.θ[:T₀]
        
    states[:film] .= 0
        
    states[:Q] .= 0

    states[:SOH] .= 1
    
    # algebraic
    states[:j] .= 0
        
    states[:Φ_e] .= 0
    
    build_c_s_star!(states, p) # needed for the OCV
    build_OCV!(states, p)
    states[:Φ_s] = states[:U]
    
    states[:I] = 0
    
    states[:j_s] .= 0

    return states
end