function residuals_PET!(residuals, t, x, ẋ, method::Symbol, p::AbstractParam;
    symbolic = true,  # if the model is being evaluated symbolically by ModelingToolkit
    )

    check_appropriate_method(method)
    
    ## First put the vector of x's, ẋ's, and residuals into dictionaries
    states = retrieve_states(x, p)
    ∂states = retrieve_states(ẋ, p)
    res = retrieve_states(residuals, p)

    states[:t] = t
    states[:method] = method
    states[:symbolic] = symbolic

    ## Calculate a few necessary auxiliary variables
    # The final value of the x vector can either be I or P,
    # so this function specifically designates the I and P states
    build_I_V_P!(states, p)

    # create a state j_aging which is a combination of j and j_s
    build_j_aging!(states, p)

    # Temperature, T
    build_T!(states, p)

    # Surface average concentration, c_s_star
    build_c_s_star!(states, p)

    # Open circuit voltages, U
    build_OCV!(states, p)

    # Overpotentials, η
    build_η!(states, p)

    # Electrolyte conductivity, K_eff
    build_K_eff!(states, p)
    
    
    ## Differential residuals
    # Residual for the electrolyte concentration, c_e
    residuals_c_e!(res, states, ∂states, p)

    # Residual for the average surface concentration, c_s_avg
    residuals_c_s_avg!(res, states, ∂states, p)
    
    # If the polynomial fit for the surface concentration is used
    if p.numerics.solid_diffusion === :polynomial
        residuals_Q!(res, states, ∂states, p)
    end

    # Ageing effects take place only during charging processes [as assumed here].
    # It is necessary to switch between the case in which the applied current density is a numerical
    # quantity & in the case in which is a symbolical quantity.
    # If the applied current is a numerical quantity, then perform the regular computations
    if p.numerics.aging === :SEI
        residuals_film!(res, states, ∂states, p)
    end

    # Check if the thermal dynamics are enabled.
    if p.numerics.temperature
        build_heat_generation_rates!(states, p)
        residuals_T!(res, states, ∂states, p)
    end


    ## Algebraic residuals
    # Residuals for ionic flux, j
    residuals_j!(res, states, p)
    
    # Residuals for ionic flux with aging, j_s
    if p.numerics.aging ∈ (:SEI, :R_film)
        residuals_j_s!(res, states, p)
    end

    # Residuals for the electrolyte potential, Φ_e
    residuals_Φ_e!(res, states, p)

    # Residuals for the solid potential, Φ_s
    residuals_Φ_s!(res, states, p)

    # Residuals for applied current density, I
    residuals_I_V_P!(res, states, p)

    ## Compile all residuals together
    build_residuals!(residuals, res, p)

    return nothing
end

function build_I_V_P!(states, p)
    """
    If the model is using current or voltage, build the voltage.
    If the model is using power, the input `I` is really the power input. The state `I` must be 
    redefined to be I = P/V
    """

    Φ_s = states[:Φ_s]

    if p.numerics.edge_values === :center
        V = Φ_s[1] - Φ_s[end]
    else # interpolated edges
        V = (1.5*Φ_s[1] - 0.5*Φ_s[2]) - (1.5*Φ_s[end] - 0.5*Φ_s[end-1])
    end
    
    states[:V] = state_new([V], (), p)

    if states[:method] ∈ (:I, :V)
        I = states[:I][1]
        P = I*V

        states[:P] = state_new([P], (), p)
    elseif states[:method] === :P
        # What is currently I should really be P. I is not defined yet
        states[:P] = states[:I]

        P = states[:P][1]
        I = P/V

        states[:I] = state_new([I], (), p)
    end

    return nothing
end

function build_j_aging!(states, p)
    """
    Append `j` with possibly additions from `j_s`
    """
    j = states[:j]
    j_s = states[:j_s]

    j_aging = p.numerics.aging ∈ (:SEI, :R_film) ? j .+ [zeros(p.N.p);j_s] : j

    states[:j_aging] = state_new(j, (:p, :n), p)

    return nothing
end

function build_T!(states, p)
    """
    If temperature is not enabled, include a vector of temperatures using the specified initial temperature.
    """
    if isempty(states[:T])
        T = repeat([p.θ[:T₀]], (p.N.p+p.N.s+p.N.n+p.N.a+p.N.z))
        
        states[:T] = state_new(T, (:a, :p, :s, :n, :z), p)
    end
    return nothing
end

function build_c_s_star!(states, p::AbstractParam)
    # build_c_s_star! evaluates the concentration of Li-ions at the electrode surfaces.
    
    c_s_avg = states[:c_s_avg]
    j = states[:j]
    Q = states[:Q]
    T = states[:T]
    
    # Check what kind of solid diffusion model has been chosen.
    if p.numerics.solid_diffusion ∈ (:quadratic, :polynomial)

        # Diffusion coefficients for the solid phase
        D_sp_eff, D_sn_eff = p.numerics.D_s_eff(c_s_avg.p, c_s_avg.n, T.p, T.n, p)
        if p.numerics.solid_diffusion === :quadratic # Two peters model
            # Evaluates the average surface concentration in both the electrodes.
            # Cathode
            c_s_star_p = c_s_avg.p-(p.θ[:Rp_p]./(D_sp_eff.*5)).*j.p
            # Anode
            c_s_star_n = c_s_avg.n-(p.θ[:Rp_n]./(D_sn_eff.*5)).*j.n
        elseif p.numerics.solid_diffusion === :polynomial # Three peters model
            
            # Cathode
            c_s_star_p = c_s_avg.p+(p.θ[:Rp_p]./(D_sp_eff.*35)).*(-j.p+8*D_sp_eff.*Q.p)
            # Anode
            c_s_star_n = c_s_avg.n+(p.θ[:Rp_n]./(D_sn_eff.*35)).*(-j.n+8*D_sn_eff.*Q.n)
        end

    elseif p.numerics.solid_diffusion === :Fickian # Full model
        p_indices = p.N.r_p:p.N.r_p:p.N.r_p*p.N.p
        n_indices = p.N.r_n:p.N.r_n:p.N.r_n*p.N.n
        
        c_s_star_p = c_s_avg.p[p_indices]
        c_s_star_n = c_s_avg.n[n_indices]
    end
    # Return the residuals
    c_s_star = [c_s_star_p; c_s_star_n]

    states[:c_s_star] = state_new(c_s_star, (:p, :n), p)

    return nothing
end

function build_OCV!(states, p)
    """
    In-place barrier function for build_OCV
    """
    c_s_star = states[:c_s_star]
    T = states[:T]

    # Compute the OCV for the positive & negative electrodes.
    U_p, U_n, ∂U∂T_p, ∂U∂T_n = build_OCV(c_s_star, T, p)

    states[:U] = state_new([U_p; U_n],       (:p, :n), p)
    states[:∂U∂T] = state_new([∂U∂T_p; ∂U∂T_n], (:p, :n), p)

    return nothing
end

@views @inbounds function build_OCV(c_s_star, T, p)
    """
    Calculate the open circuit voltages for the positive & negative electrodes
    """
    c_s_star_p = c_s_star[(1:p.N.p)]
    c_s_star_n = c_s_star[(1:p.N.n) .+ (p.N.p)]
    T_p = T[(1:p.N.p) .+ (p.N.a)]
    T_n = T[(1:p.N.n) .+ (p.N.a+p.N.p+p.N.s)]
    
    θ_p = c_s_star_p./p.θ[:c_max_p]
    θ_n = c_s_star_n./p.θ[:c_max_n]

    U_p, ∂U∂T_p = p.numerics.OCV_p(θ_p, T_p, p)
    U_n, ∂U∂T_n = p.numerics.OCV_n(θ_n, T_n, p)

    return U_p, U_n, ∂U∂T_p, ∂U∂T_n
end

function build_η!(states, p)
    """
    Calculate the overpotentials for the positive & negative electrodes
    """

    Φ_s = states[:Φ_s]
    Φ_e = states[:Φ_e]
    U = states[:U]
    j = states[:j]
    j_s = states[:j_s]
    film = states[:film]

    F = 96485.3365
    R = 8.31446261815324

    η_p = @. Φ_s.p - Φ_e.p - U.p
    η_n = @. Φ_s.n - Φ_e.n - U.n
    if     p.numerics.aging === :SEI
        η_n .+= @. - F*(j.n + j_s)*(p.θ[:R_SEI] + film/p.θ[:k_n_aging])
    elseif p.numerics.aging === :R_film
        η_n .+= @. - F*(j.n + j_s)*p.θ[:R_film]
    end

    states[:η] = state_new([η_p; η_n], (:p, :n), p)

    return nothing
end

function build_residuals!(res_tot::AbstractVector, res::Dict, p::AbstractParam)
    """
    Create the residuals vector using all the variables which are needed in the simulation
    """
    @inbounds for var in p.cache.vars
        ind_var = getproperty(p.ind, var)
        res_tot[ind_var] .= res[var]
    end
    return nothing
end

function build_K_eff!(states, p::AbstractParam)
    c_e = states[:c_e]
    T = states[:T]

    K_eff_p, K_eff_s, K_eff_n = p.numerics.K_eff(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)

    states[:K_eff] = state_new([K_eff_p; K_eff_s; K_eff_n], (:p, :s, :n), p)
    
    return nothing
end

function active_material(p::AbstractParam)
    """
    Electrode active material fraction [-]
    """
    ϵ_sp = (1.0 - p.θ[:ϵ_p] - p.θ[:ϵ_fp])
    if p.numerics.aging === :R_film
        ϵ_sn = (1.0 - p.θ[:ϵ_n][1] - p.θ[:ϵ_fn])
    else
        ϵ_sn = (1.0 - p.θ[:ϵ_n] - p.θ[:ϵ_fn])
    end

    return ϵ_sp, ϵ_sn
end

function conductivity_effective(p::AbstractParam)
    """
    Effective conductivity [S/m]
    """
    ϵ_sp, ϵ_sn = active_material(p)

    σ_eff_p = p.θ[:σ_p]*ϵ_sp
    σ_eff_n = p.θ[:σ_n]*ϵ_sn

    return σ_eff_p, σ_eff_n
end

function surface_area_to_volume_ratio(p::AbstractParam)
    """
    Surface area to volume ratio for a sphere (SA/V = 4πr^2/(4/3πr^3)) multipled by the active material fraction [m^2/m^3]
    """
    ϵ_sp, ϵ_sn = active_material(p)

    a_p = 3ϵ_sp/p.θ[:Rp_p]
    a_n = 3ϵ_sn/p.θ[:Rp_n]

    return a_p, a_n
end

function coeff_reaction_rate(states, p::AbstractParam)
    """
    Reaction rates (k) of cathode and anode [m^2.5/(m^0.5 s)]
    """
    T = states[:T]
    c_s_avg = states[:c_s_avg]

    return p.numerics.rxn_rate(T.p, T.n, c_s_avg.p, c_s_avg.n, p)

end

function residuals_Φ_s!(res, states, p::AbstractParam)
    """
    residuals_Φ_s! evaluates the residuals of the solid potential equation [V]
    """

    j = states[:j_aging]
    Φ_s = states[:Φ_s]
    I_density = states[:I][1]

    res_Φ_s = res[:Φ_s]

    F = 96485.3365
    Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)
    
    ## Positive electrode

    # RHS for the solid potential in the positive electrode. The BC on the left is enforced [Neumann BC]
    σ_eff_p, σ_eff_n = conductivity_effective(p)
    a_p, a_n = surface_area_to_volume_ratio(p)

    @views @inbounds f_p = [
        ((p.θ[:l_p].*Δx_p.*a_p.*F.*j.p[1])-I_density).*Δx_p.*p.θ[:l_p]./σ_eff_p
        (p.θ[:l_p].^2 .* Δx_p.^2 .* a_p.*F.*j.p[2:end])./σ_eff_p
    ]
    
    # RHS for the solid potential in the positive electrode.

    ## Negative electrode

    # RHS for the solid potential in the negative electrode.
    @views @inbounds f_n = (p.θ[:l_n].^2 .* Δx_n.^2 .* a_n.*F.*j.n[1:end-1])./σ_eff_n

    if states[:method] ∈ (:I, :V) # The Neumann BC on the right is enforced only when operating using applied current density as the input()
        # RHS for the solid potential in the negative electrode.
        append!(f_n, ((p.θ[:l_n].*Δx_n.*a_n.*F.*j.n[end])+I_density).*Δx_n.*p.θ[:l_n]./σ_eff_n)
    end

    function block_matrix_Φ_s(N::Int, mat_type=Float64)
        A = zeros(mat_type, N, N)
    
        ind_diagonal = diagind(A)
        ind_neg1diag = diagind(A, -1)
        ind_pos1diag = diagind(A, 1)
    
        A[ind_diagonal]    .= -2.0
        A[ind_pos1diag] .=  1.0
        A[ind_neg1diag] .=  1.0
        A[1,1] = -1.0
        A[end,end] = -1.0
    
        return A
    end

    ## Residual array
    # Return the residual array
    A_p = block_matrix_Φ_s(p.N.p, eltype(j.p))
    A_n = block_matrix_Φ_s(p.N.n, eltype(j.p))

    if states[:method] === :P # Power mode
        A_n = A_n[1:end-1,:]
        A_n *= Φ_s.n
        A_n .-= f_n

        if p.numerics.edge_values === :center
            Φ_s_pos_cc = Φ_s.p[1]
            Φ_s_neg_cc = Φ_s.n[end]
        else # 2, interpolated edges
            Φ_s_pos_cc = 1.5*Φ_s.p[1]   - 0.5*Φ_s.p[2]
            Φ_s_neg_cc = 1.5*Φ_s.n[end] - 0.5*Φ_s.n[end-1]
        end
        P = states[:P][1]

        Φ_s_n_BC = P
        Φ_s_n_BC += Φ_s_pos_cc * (σ_eff_p / (Δx_p * p.θ[:l_p]) * (Φ_s.p[2]     - Φ_s.p[1])   - p.θ[:l_p] * Δx_p * a_p * F * j.p[1])
        Φ_s_n_BC += Φ_s_neg_cc * (σ_eff_n / (Δx_n * p.θ[:l_n]) * (Φ_s.n[end-1] - Φ_s.n[end]) - p.θ[:l_n] * Δx_n * a_n * F * j.n[end])

        append!(A_n, Φ_s_n_BC)

    elseif states[:method] ∈ (:I, :V) # Current or voltage modes
        A_n *= Φ_s.n
        A_n .-= f_n

    end

    A_p *= Φ_s.p
    A_p .-= f_p

    res_Φ_s .= [A_p; A_n]

    return nothing
end

function residuals_Φ_e!(res, states, p::AbstractParam)
    """
    residuals_Φ_e! evaluates residuals for the electrolyte potential equation discretized using method of lines, [V]
    """

    j = states[:j_aging]
    Φ_e = states[:Φ_e]
    c_e = states[:c_e]
    T = states[:T]

    K_eff = states[:K_eff]

    res_Φ_e = res[:Φ_e]

    Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)
    a_p, a_n = surface_area_to_volume_ratio(p)
    
    R = 8.31446261815324
    F = 96485.3365

    # Since the values of K_eff are evaluated at the c_enter of each CV; there is the need to interpolate these quantities
    # & find their values at the edges of the CVs
    K̂_eff_p, K̂_eff_s, K̂_eff_n = interpolate_electrolyte_conductivities(K_eff.p, K_eff.s, K_eff.n, p)

    A_tot = block_matrix_maker(p, K̂_eff_p, K̂_eff_s, K̂_eff_n)

    # dividing by the length and Δx
    A_tot[(1:p.N.p), (1:p.N.p)] ./= (Δx_p*p.θ[:l_p])
    A_tot[(1:p.N.s) .+ (p.N.p), (1:p.N.s) .+ (p.N.p)] ./= (Δx_s*p.θ[:l_s])
    A_tot[(1:p.N.n) .+ (p.N.p + p.N.s), (1:p.N.n) .+ (p.N.p + p.N.s)] ./= (Δx_n*p.θ[:l_n])

    # Fix values to enforce BC on the left side of the positive electrode.
    A_tot[1,1] = K̂_eff_p[1]./(Δx_p*p.θ[:l_p])
    A_tot[1,2] = -K̂_eff_p[1]./(Δx_p*p.θ[:l_p])

    # The value of Φ_e in the last volume of the negative electrode is known
    # & fixed.
    # right now, we have -K_eff [last interior face] + K_eff (last interior
    # face)
    
    if p.numerics.edge_values === :center
        A_tot[end, end-1:end] .= [0.0, 1.0]
    elseif p.numerics.edge_values === :edge
        A_tot[end, end-1:end] .= [-1/3, 1.0]
    end
    ## Interfaces Positive electrode [last volume of the positive]

    # Here we are in the last volume of the positive
    den = (Δx_p*p.θ[:l_p]/2+Δx_s*p.θ[:l_s]/2)
    last_p = K̂_eff_p[end-1]/(Δx_p*p.θ[:l_p])
    A_tot[p.N.p,p.N.p-1:p.N.p+1] = [-last_p, (last_p+K̂_eff_p[end]/den), -K̂_eff_p[end]/den]
    ## Interfaces Positive electrode [first volume of the separator]

    # Here we are in the first volume of the separator
    first_s = K̂_eff_s[1]/(Δx_s*p.θ[:l_s])
    A_tot[p.N.p+1,p.N.p:p.N.p+2] = [-K̂_eff_p[end]/den, (first_s+K̂_eff_p[end]/den), -first_s]

    ## Interfaces Positive electrode [last volume of the separator]
    # Here we are in the last volume of the separator
    den = (Δx_n*p.θ[:l_n]/2+Δx_s*p.θ[:l_s]/2)
    last_s = K̂_eff_s[end-1]/(Δx_s*p.θ[:l_s])
    A_tot[p.N.p+p.N.s,p.N.p+p.N.s-1:p.N.p+p.N.s+1] = [-last_s, (last_s+K̂_eff_s[end]/den), -K̂_eff_s[end]/den]

    ## Interfaces Positive electrode [first volume of the negative]
    # Here we are inside the first volume of the negative electrode
    first_n = K̂_eff_n[1]/(Δx_n*p.θ[:l_n])
    A_tot[p.N.p+p.N.s+1,p.N.p+p.N.s:p.N.p+p.N.s+2] = [-K̂_eff_s[end]/den, (first_n+K̂_eff_s[end]/den), -first_n]


    ## Electrolyte concentration interpolation
    # Evaluate the interpolation of the electrolyte concentration values at the
    # edges of the control volumes.
    c̄_e_p, c̄_e_ps, c̄_e_s, c̄_e_sn, c̄_e_n = interpolate_electrolyte_concentration(c_e, p)
    ## Temperature interpolation
    # Evaluate the temperature value at the edges of the control volumes
    T̄_p, T̄_ps, T̄_s, T̄_sn, T̄_n = interpolate_temperature(T, p)
    ## Electrolyte fluxes
    # Evaluate the interpolation of the electrolyte concentration fluxes at the
    # edges of the control volumes.
    c_e_flux_p, c_e_flux_ps, c_e_flux_s, c_e_flux_sn, c_e_flux_n = interpolate_electrolyte_concetration_fluxes(c_e, p)
    ## RHS arrays
    K = 2R*(1.0 - p.θ[:t₊])/F

    prod_tot = [
        K̂_eff_p.*[T̄_p; T̄_ps].*[c_e_flux_p; c_e_flux_ps].*[1.0./c̄_e_p; 1.0./c̄_e_ps] # p
        K̂_eff_s.*[T̄_s; T̄_sn].*[c_e_flux_s; c_e_flux_sn].*[1.0./c̄_e_s; 1.0./c̄_e_sn] # s
        K̂_eff_n[1:end-1].*T̄_n.*c_e_flux_n./c̄_e_n # n
    ]
    prop_p_1 = prod_tot[1]

    prod_tot = prod_tot[2:end] .- prod_tot[1:end-1]
    prepend!(prod_tot, prop_p_1)

    f = -K*prod_tot

    ind_p = (1:p.N.p)
    ind_n = (1:p.N.n) .+ (p.N.p+p.N.s)

    f[ind_p] .+= (Δx_p*p.θ[:l_p]*F*a_p)*j.p
    f[ind_n] .+= (Δx_n*p.θ[:l_n]*F*a_n)*j.n[1:end-1]

    # Set the last element of Φ_e to 0 [enforcing BC]
    append!(f, 0.0)

    # Return the residual value for the electrolyte potential
    res_Φ_e .= A_tot*Φ_e .- f

    return nothing
end

function coeff_solid_diffusion_effective(states::Dict, p::AbstractParam)
    c_s_avg = states[:c_s_avg]
    T = states[:T]
    
    return p.numerics.D_s_eff(c_s_avg.p, c_s_avg.n, T.p, T.n, p)
end

function coeff_electrolyte_diffusion_effective(states::Dict, p::AbstractParam)
    c_e = states[:c_e]
    T = states[:T]
    
    return p.numerics.D_eff(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)
end

function residuals_c_e!(res, states, ∂states, p::AbstractParam)
    """
    Calculate the electrolyte concentration residuals
    """

    c_e = states[:c_e]
    j = states[:j]

    ∂c_e = ∂states[:c_e]

    res_c_e = res[:c_e]

    Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)

    # Diffusion coefficients
    # Comment this for benchmark purposes
    D_eff_p, D_eff_s, D_eff_n = coeff_electrolyte_diffusion_effective(states, p)
    a_p, a_n = surface_area_to_volume_ratio(p)

    # Interpolation of the diffusion coefficients, same for electrolyte conductivities
    D_eff_p, D_eff_s, D_eff_n = interpolate_electrolyte_conductivities(D_eff_p, D_eff_s, D_eff_n, p)

    A_tot = block_matrix_maker(p, -D_eff_p, -D_eff_s, -D_eff_n)

    # dividing by the length and Δx
    @views @inbounds A_tot[(1:p.N.p), (1:p.N.p)] ./= (Δx_p*p.θ[:l_p])^2
    @views @inbounds A_tot[(1:p.N.s) .+ (p.N.p), (1:p.N.s) .+ (p.N.p)] ./= (Δx_s*p.θ[:l_s])^2
    @views @inbounds A_tot[(1:p.N.n) .+ (p.N.p + p.N.s), (1:p.N.n) .+ (p.N.p + p.N.s)] ./= (Δx_n*p.θ[:l_n])^2

    # Reset values on the lines for the interfaces conditions
    @views @inbounds A_tot[p.N.p,:]         .= 0.0
    @views @inbounds A_tot[p.N.p+1,:]       .= 0.0

    # Reset values on the lines for the interfaces conditions
    @views @inbounds A_tot[p.N.p+p.N.s,:]   .= 0.0
    @views @inbounds A_tot[p.N.p+p.N.s+1,:] .= 0.0

    ## Interface between separator & positive electrode [last volume in the positive electrode]

    # Compute the common denominator at the interface
    @views @inbounds den_s = (Δx_p*p.θ[:l_p]/2 + Δx_s*p.θ[:l_s]/2)
    # Last diffusion coefficient of the positive electrode
    @views @inbounds last_p = D_eff_p[end-1]/(Δx_p*p.θ[:l_p])
    # Diffusion coefficient on the interface
    @views @inbounds first_s = D_eff_p[end]/den_s
    # Fix the values at the boundaries
    @views @inbounds A_tot[p.N.p,p.N.p-1:p.N.p+1] .= [last_p; -(last_p + first_s); first_s]/(Δx_p*p.θ[:l_p]*p.θ[:ϵ_p])

    ## Interface between separator & positive electrode [first volume in the separator]

    # First diffusion coefficient in the separator
    @views @inbounds second_s = D_eff_s[1]/(Δx_s*p.θ[:l_s])
    # Diffusion coefficient on the interface
    @views @inbounds first_s = D_eff_p[end]/den_s

    @views @inbounds A_tot[p.N.p+1,p.N.p:p.N.p+2] .= [first_s; -(first_s+second_s); second_s]/(Δx_s*p.θ[:l_s]*p.θ[:ϵ_s])

    ## Interface between separator & negative electrode [last volume in the separator]

    # Compute the common denominator at the interface
    @views @inbounds den_s = (Δx_s*p.θ[:l_s]/2 + Δx_n*p.θ[:l_n]/2)
    # Last diffusion coefficient in the separator
    @views @inbounds last_s = D_eff_s[end-1]/(Δx_s*p.θ[:l_s])
    # Diffusion coefficient on the interface
    @views @inbounds first_n = D_eff_s[end]/den_s

    @views @inbounds A_tot[p.N.p+p.N.s,p.N.p+p.N.s-1:p.N.p+p.N.s+1] = [last_s; -(last_s+first_n); first_n]/(Δx_s*p.θ[:l_s]*p.θ[:ϵ_s])

    ## Interface between separator & negative electrode [first volume in the negative electrode]

    # Compute the common denominator at the interface
    den_n = (Δx_s*p.θ[:l_s]/2 + Δx_n*p.θ[:l_n]/2)
    # First diffusion coefficient in the negative electrode
    @views @inbounds second_n = D_eff_n[1]/(Δx_n*p.θ[:l_n])
    # Diffusion coefficient on the interface
    @views @inbounds first_n = D_eff_s[end]/den_n

    if p.numerics.aging === :R_film
        @assert length(p.θ[:ϵ_n]) === p.N.n

        A_tot[p.N.p+p.N.s+1,p.N.p+p.N.s:p.N.p+p.N.s+2] = [first_n; -(first_n+second_n); second_n]/(Δx_n*p.θ[:l_n]*p.θ[:ϵ_n][1])
    else
        A_tot[p.N.p+p.N.s+1,p.N.p+p.N.s:p.N.p+p.N.s+2] = [first_n; -(first_n+second_n); second_n]/(Δx_n*p.θ[:l_n]*p.θ[:ϵ_n])
    end

    ϵ_tot = [
        ones(p.N.p).*p.θ[:ϵ_p]
        ones(p.N.s).*p.θ[:ϵ_s]
        ones(p.N.n).*p.θ[:ϵ_n]
        ]

    K = 1.0./ϵ_tot
    A_ϵ = zeros(eltype(K), (p.N.p+p.N.s+p.N.n), (p.N.p+p.N.s+p.N.n))
    ind_diagonal = diagind(A_ϵ)
    ind_neg1diag = diagind(A_ϵ, -1)
    ind_pos1diag = diagind(A_ϵ, 1)
    A_ϵ[ind_diagonal] .= K

    # Build porosities matrix
    @views @inbounds A_ϵ[ind_neg1diag] .= K[1:end-1]
    @views @inbounds A_ϵ[ind_pos1diag] .= K[1:end-1]

    @views @inbounds A_ϵ[p.N.p,p.N.p-1:p.N.p+1] .= 1.0
    @views @inbounds A_ϵ[p.N.p+1,p.N.p:p.N.p+2] .= 1.0

    @views @inbounds A_ϵ[p.N.p+p.N.s,p.N.p+p.N.s-1:p.N.p+p.N.s+1] .= 1.0
    @views @inbounds A_ϵ[p.N.p+p.N.s+1,p.N.p+p.N.s:p.N.p+p.N.s+2] .= 1.0

    A_tot .*= A_ϵ

    # Write the RHS of the equation
    rhsCe = A_tot*c_e
    
    ind_p = (1:p.N.p)
    ind_n = (1:p.N.n) .+ (p.N.p+p.N.s)

    rhsCe[ind_p] .+= K[ind_p].*(1-p.θ[:t₊]).*a_p.*j.p
    # nothing for the separator since a_s = 0
    rhsCe[ind_n] .+= K[ind_n].*(1-p.θ[:t₊]).*a_n.*j.n

    # Write the residual of the equation
    res_c_e .= rhsCe .- ∂c_e

    return nothing
end

function reaction_p(k_p, c_s_star_p, c_e_p, T_p, η_p, p::AbstractParam)
    p.numerics.rxn_p(c_s_star_p, c_e_p, T_p, η_p, k_p, p.θ[:λ_MHC_p], p.θ[:c_max_p], p)
end
function reaction_n(k_n, c_s_star_n, c_e_n, T_n, η_n, p::AbstractParam)
    p.numerics.rxn_n(c_s_star_n, c_e_n, T_n, η_n, k_n, p.θ[:λ_MHC_n], p.θ[:c_max_n], p)
end

function residuals_j!(res, states, p::AbstractParam)
    """
    Calculate the molar flux density of Li-ions residuals at the electrode-electrolyte interface [mol/(m²•s)]
    """
    c_s_star = states[:c_s_star]
    c_e = states[:c_e]
    T = states[:T]
    j = states[:j_aging]

    η = states[:η]
    U = states[:U]

    res_j = res[:j]

    # Calculate the reaction rates
    k_p_eff, k_n_eff = coeff_reaction_rate(states, p)

    j_p_calc = reaction_p(k_p_eff, c_s_star.p, c_e.p, T.p, η.p, p)
    j_n_calc = reaction_n(k_n_eff, c_s_star.n, c_e.n, T.n, η.n, p)

    res_j .= [j_p_calc; j_n_calc] .- j
    
    return nothing
end

function residuals_j_s!(res, states, p)
    """
    Calculate the anode-side molar flux density residuals due to SEI resistance [mol/(m²•s)]
    """
    j_s = states[:j_s]
    T = states[:T]
    I_density = states[:I][1]

    res_j_s = res[:j_s]

    η = states[:η]

    F = 96485.3365
    R = 8.31446261815324
    
    if p.numerics.aging === :R_film
        I1C = (F/3600.0)*p.θ[:c_max_n]*(p.θ[:θ_max_n] - p.θ[:θ_min_n])*(1.0 - p.θ[:ϵ_n][1] - p.θ[:ϵ_fn])*p.θ[:l_n]
    else
        I1C = (F/3600.0)*p.θ[:c_max_n]*(p.θ[:θ_max_n] - p.θ[:θ_min_n])*(1.0 - p.θ[:ϵ_n] - p.θ[:ϵ_fn])*p.θ[:l_n]
    end
    
    α = 0.5.*F./(R.*T.n)
    # If aging is enabled; take into account the SEI resistance
    if p.numerics.aging === :SEI
        # Tafel equation for the side reaction flux.
        j_s_calc = -p.θ[:i_0_jside].*(I_density/I1C)^p.θ[:w].*(exp.(-α.*η.n))./F
    elseif p.numerics.aging === :R_film
        # Tafel equation for the side reaction flux.
        α = 0.5.*F./(R.*T.n)
        j_s_calc = -p.θ[:i_0_jside].*(I_density/I1C)^p.θ[:w].*(exp.(-α.*η.n))./F
    end

    # side reaction residuals
    for i in 1:length(res[:j_s])
        res[:j_s][i] = ifelse(
            I_density > 0, 
            j_s[i] .- j_s_calc[i],
            j_s[i],
        )
    end
    
    return nothing
end

function residuals_I_V_P!(res, states, p::AbstractParam)
    """
    *** THE RESIDUALS ARE HANDLED IN `fix_res!` ***

    Calculate the final scalar residuals [A/m²]

    In `fix_res!`, the residual here is subtracted from a known value,
    e.g, for current this residual is exactly 0 because the current is 
    specified exactly. For voltage, however, this equation will not be 
    exactly 0: it's `res_V = f(states) - CV`. The reason that these
    equations are here is to ensure that the Jacobian has the proper
    structure even though they are unused in the actual residuals function.
    """

    I = states[:I][1]
    V = states[:V][1]
    P = states[:P][1]
    
    res_I = res[:I]

    if     states[:method] === :I
        res_I .= I # - input value in `fix_res!`
    elseif states[:method] === :V
        res_I .= V # - input value in `fix_res!`
    elseif states[:method] === :P
        res_I .= P # - input value in `fix_res!`
    end
    
    return nothing
end

function residuals_c_s_avg_Fickian_FDM!(res, states, ∂states, p)
    """
    Calculate the volume-averaged solid particle concentration residuals using a 9th order accurate finite difference method (FDM) [mol/m³]
    """
    j = states[:j]
    c_s_avg = states[:c_s_avg]

    ∂c_s_avg = ∂states[:c_s_avg]

    res_c_s_avg = res[:c_s_avg]

    # Matrices needed for first and second order derivatives
    FO_D_p, FO_D_c_p, SO_D_p, SO_D_c_p, SO_D_Δx_p = derivative_matrices_first_and_second_order(p.N.r_p)
    FO_D_n, FO_D_c_n, SO_D_n, SO_D_c_n, SO_D_Δx_n = derivative_matrices_first_and_second_order(p.N.r_n)

    # If the regular diffusion equation is selected; then use FDM to
    # evaluate the complete solution.

    # First; retreive the diffusion coefficients
    D_sp_eff, D_sn_eff = coeff_solid_diffusion_effective(states, p)
    # Initialize the variables
    rhsCs_p = eltype(j)[]
    rhsCs_n = eltype(j)[]

    # For every single CV in the cathode; let assume the presence of a
    # solid particle
    
    @inbounds for i = 1:p.N.p
        c_s = c_s_avg.p[(i-1)*p.N.r_p+1:i*p.N.r_p]
        
        # Evaluate first order derivatives
        ∂ₓc_s_avg_p = FO_D_c_p*(FO_D_p*c_s)

        # Impose the BCs [the radial direction is normalized between 0 & 1 for improving the numerical robustness]
        # r = 1
        ∂ₓc_s_avg_p[p.N.r_p] = -j.p[i]/D_sp_eff[i]*p.θ[:Rp_p]

        # r = 0 using l'Hopital rule
        ∂ₓc_s_avg_p[1] = 0

        # Second order derivatives. The division by 6 comes from the
        # multiplication by 6 of the original differentiation matrix. This
        # reduces the presence of numerical errors.
        ∂ₓₓc_s_avg_p = SO_D_c_p/6*(SO_D_p*c_s)

        # In both r = 0 & r = 1 Neumann BCs are required. For r=0; this
        # process is not carried out since it is required that cs'(r=0) = 0.
        # For r=1 the following modification is performed (according to the
        # particular numerical scheme here adopted for the approximation of the
        # second order derivative)
        ∂ₓₓc_s_avg_p[end] += 50*SO_D_Δx_p*∂ₓc_s_avg_p[p.N.r_p]*SO_D_c_p

        # Create rhs arrays & residuals
        RHS = (D_sp_eff[i]./p.θ[:Rp_p]^2) .* [
            3∂ₓₓc_s_avg_p[1]
            ∂ₓₓc_s_avg_p[2:end]+2.0./range(1/(p.N.r_p-1), 1, length=p.N.r_p-1).*∂ₓc_s_avg_p[2:end]
            ]
        append!(rhsCs_p, RHS)

    end

    @inbounds for i=1:p.N.n
        c_s = c_s_avg.n[(i-1)*p.N.r_n+1:i*p.N.r_n]
        
        # Evaluate first order derivatives
        ∂ₓc_s_avg_n = FO_D_c_n*(FO_D_n*c_s)

        # Impose the BCs [the radial direction is normalized between 0 & 1 for improving the numerical robustness]
        # r = 1
        ∂ₓc_s_avg_n[p.N.r_n] = -j.n[i]/D_sn_eff[i]*p.θ[:Rp_n]

        # r = 0 using l'Hopital rule
        ∂ₓc_s_avg_n[1] = 0

        # Second order derivatives. The division by 6 comes from the
        # multiplication by 6 of the original differentiation matrix. This
        # reduces the presence of numerical errors.
        ∂ₓₓc_s_avg_n = SO_D_c_n/6*(SO_D_n*c_s)

        # In both r = 0 & r = 1 Neumann BCs are required. For r=0; this
        # process is not carried out since it is required that cs'(r=0) = 0.
        # For r=1 the following modification is performed (according to the
        # particular numerical scheme here adopted for the approximation of the
        # second order derivative)
        ∂ₓₓc_s_avg_n[end] += 50*SO_D_Δx_n*∂ₓc_s_avg_n[p.N.r_n]*SO_D_c_n

        # Create rhs arrays & residuals
        RHS = D_sn_eff[i]./p.θ[:Rp_n]^2 .* [
            3∂ₓₓc_s_avg_n[1]
            ∂ₓₓc_s_avg_n[2:end]+2.0./(range(1/(p.N.r_n-1), 1.0, length=(p.N.r_n-1))).*∂ₓc_s_avg_n[2:end]
        ]

        append!(rhsCs_n, RHS)
    end
    
    res_c_s_avg .= [rhsCs_p; rhsCs_n] .- ∂c_s_avg

    return nothing
end

function residuals_c_s_avg_Fickian_spectral!(res, states, ∂states, p)
    """
    Calculate the volume-averaged solid particle concentration residuals using a spectral method [mol/m³]
    """
    j = states[:j]
    j_p = states[:j]
    c_s_avg_n = states[:c_s_avg]

    ∂c_s_avg = ∂states[:c_s_avg]

    res_c_s_avg = res[:c_s_avg]

    function cheb(N)
        if iszero(N)
            return 0, 1
        end

        x = cos.(pi*(0:N)/N)
        c = [2.0; ones(N-1); 2.0].*(-1).^(0:N)

        X = repeat(x,1,N+1)
        dX = X .- X'
        D = (c*(1.0./c)')./(dX .+ I(N+1)) # off-diagonal entries
        D[diagind(D)]  .-= sum(D', dims=1)[:]

        return D, x
    end
    
    flip(x) = x[end:-1:1]
    
    # First, retreive the diffusion coefficients
    D_sp_eff, D_sn_eff = coeff_solid_diffusion_effective(states, p)
    
    # Initialize the variables
    rhsCs_p = eltype(j_p)[]
    rhsCs_n = eltype(j_p)[]

    DiffMat_p, Rad_position_p = cheb(p.N.r_p-1)
    DiffMat_n, Rad_position_n = cheb(p.N.r_n-1)

    # For every single CV in the cathode, let assume the presence of a solid particle
    for i in 1:p.N.p
        # LIONSIMBA scheme vector: 1 (top) is particle centre, end(bottom) - particle surface
        c_s = c_s_avg.p[(i-1)*p.N.r_p+1:i*p.N.r_p]
        
        c_s_flipped = flip(c_s)      # cheb matrices work on [1 to -1] ordering
        ∂ₓc_s_avg_p = DiffMat_p*c_s_flipped
        ∂ₓc_s_avg_p[1] = -j.p[i]*p.θ[:Rp_p]*0.5/D_sp_eff[i] # modified BC value due to cheb scheme
        ∂ₓc_s_avg_p[end] = 0
        
        # Below line: we compute the RHS of the Fick's law PDE as it appears in LIONSIMBA
        # paper, but in scaled co-ordinate system. Note that this is missing a leading
        # scaling factor (1/(r+1)^2),which will be included in later lines of code in
        # this file At the centre, the equation becomes degenerate and hence treated
        # separately. The scaling factor is included at a later point below.
        
        rhs_numerator_p = DiffMat_p*(4*D_sp_eff[i]*((Rad_position_p .+ 1).^2).*∂ₓc_s_avg_p/(p.θ[:Rp_p]^2))
        
        rhs_limit_vector = (4*D_sp_eff[i]/p.θ[:Rp_p]^2)*3*(DiffMat_p*∂ₓc_s_avg_p) # limit at r_tilde tends to -1 (at centre)
        ∂ₓc_s_avg_p = flip(∂ₓc_s_avg_p) # flip back to be compatible with LIONSIMBA
        rhs_numerator_p = flip(rhs_numerator_p)
        
        append!(rhsCs_p, rhs_limit_vector[end]) # clever trick to apply the L'hopital's rule at particle centre
        append!(rhsCs_p, rhs_numerator_p[2:end]./((Rad_position_p[end-1:-1:1] .+ 1).^2)) # Apply the scaling factor for the Fick's law RHS for the rest of the shells excluding particle centre
    end

    for i=1:p.N.n
        # Lionsimba scheme vector: 1 (top) is particle centre, end(bottom) - particle surface
        c_s = c_s_avg.n[(i-1)*p.N.r_n+1:i*p.N.r_n]
        
        c_s_flipped = flip(c_s)      # cheb matrices work with [1 to -1] ordering
        ∂ₓc_s_avg_n = DiffMat_n*c_s_flipped
        ∂ₓc_s_avg_n[1] = -j.n[i]*p.θ[:Rp_n]*0.5/D_sn_eff[i] # modified BC value due to cheb scheme
        ∂ₓc_s_avg_n[end] = 0
        
        # Below line: we compute the RHS of the Fick's law PDE as it appears in LIONSIMBA
        # paper, but in scaled co-ordinate system. Note that this is missing a leading
        # scaling factor (1/(r+1)^2) which will be accounted for in later lines of code
        # in this file, At the centre, the equation becomes degenerate and hence treated
        # separately. The scaling factor is included at a later point below.
        
        rhs_numerator_n = DiffMat_n*(4*D_sn_eff[i]*((Rad_position_n .+ 1).^2).*∂ₓc_s_avg_n/(p.θ[:Rp_n]^2))
        
        rhs_limit_vector = (4*D_sn_eff[i]/p.θ[:Rp_n]^2)*3*(DiffMat_n*∂ₓc_s_avg_n) # limit at r_tilde tends to -1 (at centre)
        ∂ₓc_s_avg_n = flip(∂ₓc_s_avg_n) # flip back to be compatible with LIONSIMBA
        rhs_numerator_n = flip(rhs_numerator_n)
        
        append!(rhsCs_n, rhs_limit_vector[end]) # clever trick to apply the L'hopital's rule at particle centre 'end' is used because of cpu time needed for flip
        append!(rhsCs_n, rhs_numerator_n[2:end]./((Rad_position_n[end-1:-1:1] .+ 1).^2)) # Apply the scaling factor for the Fick's law RHS for the rest of the shells excluding particle centre
    end
    
    res_c_s_avg .= [rhsCs_p; rhsCs_n] .- ∂c_s_avg
    
    return nothing
end

function residuals_c_s_avg!(res, states, ∂states, p)
    """
    Calculate the solid particle concentration residuals using various methods defined by p.numerics.solid_diffusion [mol/m³]
    """

    if p.numerics.solid_diffusion ∈ (:quadratic, :polynomial)
        j = states[:j]
    
        ∂c_s_avg = ∂states[:c_s_avg]
    
        res_c_s_avg = res[:c_s_avg]
        
        # Cathode
        rhsCs_p = -3j.p/p.θ[:Rp_p]

        # Anode
        rhsCs_n = -3j.n/p.θ[:Rp_n]

        res_c_s_avg .= [rhsCs_p; rhsCs_n] .- ∂c_s_avg
    else
        if     p.numerics.Fickian_method === :finite_difference # Use the FDM method for the solid phase diffusion
            residuals_c_s_avg_Fickian_FDM!(res, states, ∂states, p)
        elseif p.numerics.Fickian_method === :spectral # Use the spectral method for the discretization of the solid phase diffusion
            residuals_c_s_avg_Fickian_spectral!(res, states, ∂states, p)
        end
    end
    return nothing
end

function residuals_Q!(res, states, ∂states, p)
    """
    residuals_Q! is used to implement the three parameters reduced model for solid phase diffusion [mol/m⁴]
    This model has been taken from the paper, "Efficient Macro-Micro Scale Coupled
    Modeling of Batteries" - Subramanian, Diwakar, Tapriyal - 2005 JES
    """

    Q = states[:Q]
    j = states[:j]

    ∂Q = ∂states[:Q]

    res_Q = res[:Q]
    
    # Diffusion coefficients for the solid phase
    D_sp_eff, D_sn_eff = coeff_solid_diffusion_effective(states, p)
    
    rhsQ_p = @. (-30D_sp_eff*Q.p - 45/2*j.p)/p.θ[:Rp_p]^2
    rhsQ_n = @. (-30D_sn_eff*Q.n - 45/2*j.n)/p.θ[:Rp_n]^2
    
    res_Q .= [rhsQ_p; rhsQ_n] .- ∂Q
    
    return nothing
end

function residuals_film!(res, states, ∂states, p)
    """
    residuals_film! describes the dynamics of the solid-electrolyte layer at the anode side
    """

    j_s = states[:j_s]

    ∂film = ∂states[:film]
    
    res_film = res[:film]
    
    rhs_film = -j_s.*p.θ[:M_n]./p.θ[:ρ_n]

    res_film .= rhs_film .- ∂film

    return nothing
end

function residuals_T!(res, states, ∂states, p)
    """
    Calculate the 1D temperature residuals
    """
    c_e = states[:c_e]
    Φ_e = states[:Φ_e]
    Φ_s = states[:Φ_s]
    j = states[:j]
    T = states[:T]
    I_density = states[:I][1]

    Q_rev = states[:Q_rev]
    Q_rxn = states[:Q_rxn]
    Q_ohm = states[:Q_ohm]

    ∂T = ∂states[:T]

    res_T = res[:T]

    Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)
    T_ref = 298.15

    T_BC_sx =  p.θ[:h_cell]*(T_ref-T[1])/(Δx_a*p.θ[:l_a])
    T_BC_dx = -p.θ[:h_cell]*(T[end]-T_ref)/(Δx_z*p.θ[:l_z])

    # Evaluate the derivatives used in Q_ohm calculations
    function block_matrix_T(λ, N)
        A_tot = zeros(eltype(λ), N, N)
    
        ind_diagonal = diagind(A_tot)
        ind_neg1diag = diagind(A_tot, -1)
        ind_pos1diag = diagind(A_tot, 1)
    
        A_tot[ind_diagonal[1:N]]      .= -λ
        A_tot[ind_diagonal[2:N]]     .-= λ
        A_tot[ind_neg1diag[1:N-1]] .= λ
        A_tot[ind_pos1diag[1:N-1]] .= λ
    
        return A_tot
    end

    # Positive current collector
    A_a = block_matrix_T(p.θ[:λ_a], p.N.a)

    # Positive electrode
    A_p = block_matrix_T(p.θ[:λ_p], p.N.p)

    # Separator
    A_s = block_matrix_T(p.θ[:λ_s], p.N.s)

    # Negative electrode
    A_n = block_matrix_T(p.θ[:λ_n], p.N.n)

    # Negative current collector
    A_z = block_matrix_T(p.θ[:λ_z], p.N.z)

    A_z[end, end-1:end] = [p.θ[:λ_z] -p.θ[:λ_z]]

    # Divide the matrices by (Δx*l)^2

    A_a /= (Δx_a*p.θ[:l_a])^2
    A_p /= (Δx_p*p.θ[:l_p])^2
    A_s /= (Δx_s*p.θ[:l_s])^2
    A_n /= (Δx_n*p.θ[:l_n])^2
    A_z /= (Δx_z*p.θ[:l_z])^2

    A_tot = zeros(eltype(A_a), (p.N.p+p.N.s+p.N.n+p.N.a+p.N.z), (p.N.p+p.N.s+p.N.n+p.N.a+p.N.z))

    ind = 1:p.N.a
    A_tot[ind,ind] = A_a

    ind = (1:p.N.p) .+ ind[end]
    A_tot[ind,ind] = A_p

    ind = (1:p.N.s) .+ ind[end]
    A_tot[ind,ind] = A_s

    ind = (1:p.N.n) .+ ind[end]
    A_tot[ind,ind] = A_n

    ind = (1:p.N.z) .+ ind[end]
    A_tot[ind,ind] = A_z

    ## Interfaces

    # Interface between aluminium current collector & positive electrode. We
    # are in the last volume of the current collector
    β_a_p = (Δx_a*p.θ[:l_a]/2)/(Δx_a*p.θ[:l_a]/2+Δx_p*p.θ[:l_p]/2)
    λ_a_p = p.θ[:λ_a] * p.θ[:λ_p] /(β_a_p*p.θ[:λ_p] + (1-β_a_p)*p.θ[:λ_a])
    den_a_p = Δx_p*p.θ[:l_p]/2 +Δx_a*p.θ[:l_a]/2
    last_a = p.θ[:λ_a] / (Δx_a*p.θ[:l_a])
    first_p = λ_a_p/den_a_p

    A_tot[p.N.a,p.N.a-1:p.N.a+1] = [last_a -(last_a+first_p) first_p]/(Δx_a*p.θ[:l_a])

    # Interface between aluminium current collector & positive electrode. We
    # are in the first volume of the positive electrode

    den_a_p = Δx_p*p.θ[:l_p]/2 +Δx_a*p.θ[:l_a]/2
    second_p = p.θ[:λ_p] / (Δx_p*p.θ[:l_p])
    first_p = λ_a_p/den_a_p

    A_tot[p.N.a+1,p.N.a:p.N.a+2] = [first_p -(second_p+first_p) second_p]/(Δx_p*p.θ[:l_p])

    # Interface between positive electrode & separator. We
    # are in the last volume of the positive electrode
    β_p_s = (Δx_p*p.θ[:l_p]/2)/(Δx_s*p.θ[:l_s]/2+Δx_p*p.θ[:l_p]/2)
    λ_p_s = p.θ[:λ_s] * p.θ[:λ_p] /(β_p_s*p.θ[:λ_s] + (1-β_p_s)*p.θ[:λ_p])

    den_p_s = Δx_p*p.θ[:l_p]/2 +Δx_s*p.θ[:l_s]/2
    last_p = p.θ[:λ_p] / (Δx_p*p.θ[:l_p])
    first_s = λ_p_s/den_p_s

    A_tot[p.N.a+p.N.p,p.N.a+p.N.p-1:p.N.a+p.N.p+1] = [last_p -(last_p+first_s) first_s]/(Δx_p*p.θ[:l_p])

    # Interface between positive electrode & separator. We
    # are in the first volume of the separator
    den_p_s = Δx_p*p.θ[:l_p]/2 +Δx_s*p.θ[:l_s]/2
    second_s = p.θ[:λ_s] / (Δx_s*p.θ[:l_s])
    first_s = λ_p_s/den_p_s

    A_tot[p.N.a+p.N.p+1,p.N.a+p.N.p:p.N.a+p.N.p+2] = [first_s -(second_s+first_s) second_s]/(Δx_s*p.θ[:l_s])

    # Interface between separator negative electrode. We
    # are in the last volume of the separator
    β_s_n = (Δx_s*p.θ[:l_s]/2)/(Δx_s*p.θ[:l_s]/2+Δx_n*p.θ[:l_n]/2)
    λ_s_n = p.θ[:λ_s] * p.θ[:λ_n] /(β_s_n*p.θ[:λ_n] + (1-β_s_n)*p.θ[:λ_s])

    den_s_n = Δx_n*p.θ[:l_n]/2 +Δx_s*p.θ[:l_s]/2
    last_s = p.θ[:λ_s] / (Δx_s*p.θ[:l_s])
    first_n = λ_s_n/den_s_n

    A_tot[p.N.a+p.N.p+p.N.s,p.N.a+p.N.p+p.N.s-1:p.N.a+p.N.p+p.N.s+1] = [last_s -(last_s+first_n) first_n]/(Δx_s*p.θ[:l_s])

    # Interface between separator negative electrode. We
    # are in the first volume of the negative electrode

    den_s_n = Δx_n*p.θ[:l_n]/2 +Δx_s*p.θ[:l_s]/2
    second_n = p.θ[:λ_n] / (Δx_n*p.θ[:l_n])
    first_n = λ_s_n/den_s_n

    A_tot[p.N.a+p.N.p+p.N.s+1,p.N.a+p.N.p+p.N.s:p.N.a+p.N.p+p.N.s+2] = [first_n -(first_n+second_n) second_n]/(Δx_n*p.θ[:l_n])


    # Interface between negative electrode & negative current collector. We
    # are in the last volume of the negative electrode
    β_n_co = (Δx_n*p.θ[:l_n]/2)/(Δx_z*p.θ[:l_z]/2+Δx_n*p.θ[:l_n]/2)
    λ_n_co = p.θ[:λ_z] * p.θ[:λ_n] /(β_n_co*p.θ[:λ_z] + (1-β_n_co)*p.θ[:λ_n])

    den_n_co = Δx_n*p.θ[:l_n]/2 +Δx_z*p.θ[:l_z]/2
    last_n = p.θ[:λ_n] / (Δx_n*p.θ[:l_n])
    first_co = λ_n_co/den_n_co

    A_tot[p.N.a+p.N.p+p.N.s+p.N.n,p.N.a+p.N.p+p.N.s+p.N.n-1:p.N.a+p.N.p+p.N.s+p.N.n+1] = [last_n -(last_n+first_co) first_co]/(Δx_n*p.θ[:l_n])


    # Interface between negative electrode & negative current collector. We
    # are in the first volume of the negative current collector

    den_n_co = Δx_n*p.θ[:l_n]/2 +Δx_z*p.θ[:l_z]/2
    second_co = p.θ[:λ_z] / (Δx_z*p.θ[:l_z])
    first_co = λ_n_co/den_n_co

    A_tot[p.N.a+p.N.p+p.N.s+p.N.n+1,p.N.a+p.N.p+p.N.s+p.N.n:p.N.a+p.N.p+p.N.s+p.N.n+2] = [first_co -(second_co+first_co) second_co]/(Δx_z*p.θ[:l_z])

    Q_rev_tot = [
        zeros(p.N.a)
        Q_rev.p
        zeros(p.N.s)
        Q_rev.n
        zeros(p.N.z)
        ]

    Q_rxn_tot = [
        zeros(p.N.a)
        Q_rxn.p
        zeros(p.N.s)
        Q_rxn.n
        zeros(p.N.z)
        ]

    Q_ohm_tot = [
        (I_density^2)./repeat([p.θ[:σ_a]], p.N.a)
        Q_ohm.p
        Q_ohm.s
        Q_ohm.n
        (I_density^2)./repeat([p.θ[:σ_z]], p.N.z)
        ]

    BC = [
        T_BC_sx
        zeros(length(T)-2)
        T_BC_dx
        ]

    ρ_Cp = [
        repeat([p.θ[:ρ_a]*p.θ[:Cp_a]], p.N.a)
        repeat([p.θ[:ρ_p]*p.θ[:Cp_p]], p.N.p)
        repeat([p.θ[:ρ_s]*p.θ[:Cp_s]], p.N.s)
        repeat([p.θ[:ρ_n]*p.θ[:Cp_n]], p.N.n)
        repeat([p.θ[:ρ_z]*p.θ[:Cp_z]], p.N.z)
        ]

    rhsT = A_tot*T
    rhsT .+= Q_rev_tot
    rhsT .+= Q_rxn_tot
    rhsT .+= Q_ohm_tot
    rhsT .+= BC
    rhsT ./= ρ_Cp
    
    res_T .= rhsT .- ∂T
    
    return nothing
end

@inline function build_heat_generation_rates!(states, p::AbstractParam)
    """
    Evaluates the heat source terms used in the thermal model per section
    """

    Φ_s = states[:Φ_s]
    Φ_e = states[:Φ_e]
    j = states[:j]
    T = states[:T]
    c_e = states[:c_e]
    U = states[:U]
    ∂U∂T = states[:∂U∂T]
    η = states[:η]
    K_eff = states[:K_eff]

    F = 96485.3365
    R = 8.31446261815324

    # Retrieve effective electrolyte conductivity coefficients.
    a_p, a_n = surface_area_to_volume_ratio(p)
    σ_eff_p, σ_eff_n = conductivity_effective(p)

    # Evaluate the derivatives used in Q_ohm calculations
    function build_thermal_derivatives()
        # For each of the numerical derivatives computed below; the first & last control volumes are evaluated with first
        # order accuracy [forward & backward difference schemes respectively]
        # while the middle control volume approximations use a second order accuracy [central difference scheme].
    
        Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)
    
        ## Solid potential derivatives
        function stencil(N)
            A = zeros(N,N)
            A[diagind(N,N,-1)] .= -1 # central
            A[diagind(N,N,+1)] .= +1 # central

            A[1,1:3] .= [-3, 4, -1]        # forward
            A[end,end-2:end] .= [1, -4, 3] # backward

            return A
        end

        # Positive Electrode
        dΦ_sp = stencil(p.N.p)*Φ_s.p./(2*Δx_p*p.θ[:l_p])
    
        # Negative Electrode
        dΦ_sn = stencil(p.N.n)*Φ_s.n./(2*Δx_n*p.θ[:l_n])
    
        dΦ_s = [
            dΦ_sp
            dΦ_sn
            ]
    
        ## Electrolyte potential derivatives
    
        # Positive Electrode
        dΦ_ep = stencil(p.N.p)*Φ_e.p./(2*Δx_p*p.θ[:l_p])
    
        # Attention! The last volume of the positive electrode will involve one volume of the
        # separator for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # Last CV in the positive electrode: derivative approximation with a central scheme
        dΦ_ep[end] = 2*(Φ_e.s[1]-Φ_e.p[end-1])/(3 * Δx_p*p.θ[:l_p] + Δx_s*p.θ[:l_s])
    
        # Separator
    
        # Attention! The first volume of the separator will involve one volume of the
        # positive section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
        
        # Central difference scheme
        dΦ_es = stencil(p.N.s)*Φ_e.s./(2*Δx_s*p.θ[:l_s])
        
        # First CV in the separator: derivative approximation with a central difference scheme
        dΦ_es[1] = 2*(Φ_e.s[2]-Φ_e.p[end])/(Δx_p*p.θ[:l_p] + 3* Δx_s*p.θ[:l_s])
    
        # Attention! The last volume of the separator will involve one volume of the
        # negative section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # Last CV in the separator: derivative approximation with a central scheme
        dΦ_es[end] = 2*(Φ_e.n[1]-Φ_e.s[end-1])/( Δx_n*p.θ[:l_n] + 3*Δx_s*p.θ[:l_s])
    
        # Negative electrode
    
        # Attention! The first volume of the negative electrode will involve one volume of the
        # separator section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
        
        # Central difference scheme
        dΦ_en = stencil(p.N.n)*Φ_e.n./(2*Δx_n*p.θ[:l_n])
        
        # First CV in the negative electrode: derivative approximation with a central scheme
        dΦ_en[1] = 2*(Φ_e.n[2]-Φ_e.s[end])/(3 * Δx_n*p.θ[:l_n] + Δx_s*p.θ[:l_s])
        
        dΦ_e = [
            dΦ_ep
            dΦ_es
            dΦ_en
        ]
    
        ## Electrolyte concentration derivatives
    
        # Positive Electrode
        dc_ep = stencil(p.N.p)*c_e.p./(2*Δx_p*p.θ[:l_p])
    
        # Attention! The last volume of the positive electrode will involve one volume of the
        # separator for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # Last CV in the positive electrode: derivative approximation with a central scheme
        dc_ep[end] = 2*(c_e.s[1]-c_e.p[end-1])/(3 * Δx_p*p.θ[:l_p] + Δx_s*p.θ[:l_s])
    
        # Separator
    
        # Attention! The first volume of the separator will involve one volume of the
        # positive section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
        
        # Central differentiation scheme
        dc_es = stencil(p.N.s)*c_e.s./(2*Δx_s*p.θ[:l_s])
        
        # First CV in the separator: derivative approximation with a central scheme
        dc_es[1] = 2*(c_e.s[2]-c_e.p[end])/(3* Δx_s*p.θ[:l_s] + Δx_p*p.θ[:l_p])
    
        # Attention! The last volume of the separator will involve one volume of the
        # negative section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # Last CV in the separator: derivative approximation with a central scheme
        dc_es[end] = 2*(c_e.n[1]-c_e.s[end-1])/( Δx_n*p.θ[:l_n] + 3*Δx_s*p.θ[:l_s])
    
        # Negative electrode
    
        # Attention! The first volume of the negative electrode will involve one volume of the
        # separator section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        dc_en = stencil(p.N.n)*c_e.n./(2*Δx_n*p.θ[:l_n])
        
        # First CV in the negative electrode: derivative approximation with a central scheme
        dc_en[1] = 2*(c_e.n[2]-c_e.s[end])/(3 * Δx_n*p.θ[:l_n] + Δx_s*p.θ[:l_s])
    
        dc_e = [
            dc_ep
            dc_es
            dc_en
        ]

        dΦ_s = state_new(dΦ_s, (:p, :n), p)
        dΦ_e = state_new(dΦ_e, (:p, :s, :n), p)
        dc_e = state_new(dc_e, (:p, :s, :n), p)
    
        return dΦ_s, dΦ_e, dc_e
    end

    dΦ_s, dΦ_e, dc_e = build_thermal_derivatives()

    ## Reversible heat generation rate

    # Positive electrode
    Q_rev_p = @. F*a_p*j.p*T.p*∂U∂T.p

    # Negative Electrode
    Q_rev_n = @. F*a_n*j.n*T.n*∂U∂T.n

    ## Reaction heat generation rate
    @views @inbounds Q_rxn_p = @. F*a_p*j.p.*η.p
    @views @inbounds Q_rxn_n = @. F*a_n*j.n.*η.n

    ## Ohmic heat generation rate
    # Positive electrode ohmic generation rate
    Q_ohm_p = σ_eff_p * (dΦ_s.p).^2 + K_eff.p.*(dΦ_e.p).^2 + 2*R*K_eff.p.*T.p*(1-p.θ[:t₊])/F.*dc_e.p.*1.0./c_e.p.*dΦ_e.p
    # Separator ohmic generation rate
    Q_ohm_s =                         K_eff.s.*(dΦ_e.s).^2 + 2*R*K_eff.s.*T.s*(1-p.θ[:t₊])/F.*dc_e.s.*1.0./c_e.s.*dΦ_e.s
    # Negative electrode ohmic generation rate
    Q_ohm_n = σ_eff_n * (dΦ_s.n).^2 + K_eff.n.*(dΦ_e.n).^2 + 2*R*K_eff.n.*T.n*(1-p.θ[:t₊])/F.*dc_e.n.*1.0./c_e.n.*dΦ_e.n

    Q_rev = [Q_rev_p; Q_rev_n]
    Q_rxn = [Q_rxn_p; Q_rxn_n]
    Q_ohm = [Q_ohm_p; Q_ohm_s; Q_ohm_n]

    states[:Q_rev] = state_new(Q_rev, (:p, :n), p)
    states[:Q_rxn] = state_new(Q_rxn, (:p, :n), p)
    states[:Q_ohm] = state_new(Q_ohm, (:p, :s, :n), p)

    return nothing
end



@inline function calc_I1C(p::param)
    F = 96485.3365
    θ = p.θ

    if p.numerics.aging === :R_film
        @inbounds @views I1C = (F/3600.0)*min(
            θ[:c_max_n]::Float64*(θ[:θ_max_n] - θ[:θ_min_n]::Float64)*(1.0 - (θ[:ϵ_n][1])::Float64 - θ[:ϵ_fn]::Float64)*θ[:l_n]::Float64,
            θ[:c_max_p]::Float64*(θ[:θ_min_p] - θ[:θ_max_p]::Float64)*(1.0 - θ[:ϵ_p]::Float64      - θ[:ϵ_fp]::Float64)*θ[:l_p]::Float64,
        )
    else
        @inbounds @views I1C = (F/3600.0)*min(
            θ[:c_max_n]::Float64*(θ[:θ_max_n] - θ[:θ_min_n]::Float64)*(1.0 - θ[:ϵ_n]::Float64 - θ[:ϵ_fn]::Float64)*θ[:l_n]::Float64,
            θ[:c_max_p]::Float64*(θ[:θ_min_p] - θ[:θ_max_p]::Float64)*(1.0 - θ[:ϵ_p]::Float64 - θ[:ϵ_fp]::Float64)*θ[:l_p]::Float64,
        )
    end

    return I1C
end

@inline function calc_V(Y::Vector{Float64}, p::param, run::AbstractRun, ind_Φ_s::T=p.ind.Φ_s) where {T<:AbstractUnitRange{Int64}}
    if run.method === :V
        V = value(run)
    else
        if p.numerics.edge_values === :center
            V = @views @inbounds Y[ind_Φ_s[1]] - Y[ind_Φ_s[end]]
        else # interpolated edges
            V = @views @inbounds (1.5*Y[ind_Φ_s[1]] - 0.5*Y[ind_Φ_s[2]]) - (1.5*Y[ind_Φ_s[end]] - 0.5*Y[ind_Φ_s[end-1]])
        end
    end
    return V
end

@inline function calc_I(Y::Vector{Float64}, model::model_output, run::AbstractRun, p::param)
    if run.method === :I
        I = @inbounds value(run)
    elseif  run.method === :V
        I = @inbounds Y[p.ind.I[1]]
    elseif run.method === :P
        I = @inbounds value(run)/model.V[end]
    end
    
    return I
end

@inline function calc_P(Y::Vector{Float64}, model::model_output, run::AbstractRun, p::param)
    if run.method === :I || run.method === :V
        P = @views @inbounds model.I[end]*model.V[end]
    elseif run.method === :P
        P = value(run)
    end

    return P
end

@inline function calc_SOC(c_s_avg::Vector{Float64}, p::param)
    if p.numerics.solid_diffusion === :Fickian
        c_s_avg_sum = @views @inbounds mean(c_s_avg[(p.N.p*p.N.r_p)+1:end])
    else # c_s_avg in neg electrode
        c_s_avg_sum = @views @inbounds mean(c_s_avg[p.N.p+1:end])
    end

    return (c_s_avg_sum/p.θ[:c_max_n]::Float64 - p.θ[:θ_min_n]::Float64)/(p.θ[:θ_max_n]::Float64 - p.θ[:θ_min_n]::Float64) # cell-soc fraction
end