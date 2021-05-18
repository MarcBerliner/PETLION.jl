function build_auxiliary_states!(states, p::AbstractParam)
    """
    Calculates necessary auxiliary states and adds them to the `state` dict
    """

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
    
    return nothing
end

function build_I_V_P!(states, p::AbstractParam)
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

function build_j_aging!(states, p::AbstractParam)
    """
    Append `j` with possibly additions from `j_s`
    """
    j = states[:j]
    j_s = states[:j_s]

    j_aging = copy(j)
    if p.numerics.aging ∈ (:SEI, :R_film)
        j_aging[p.N.p+1:end] = j_aging[p.N.p+1:end] .+ j_s
    end

    states[:j_aging] = state_new(j, (:p, :n), p)

    return nothing
end

function build_T!(states, p::AbstractParam)
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
    """
    Evaluates the concentration of Li-ions at the electrode surfaces.
    """
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

    # Fickian diffusion
    elseif p.numerics.solid_diffusion === :Fickian
        p_indices = p.N.r_p:p.N.r_p:p.N.r_p*p.N.p
        n_indices = p.N.r_n:p.N.r_n:p.N.r_n*p.N.n
        
        c_s_star_p = @views @inbounds c_s_avg.p[p_indices]
        c_s_star_n = @views @inbounds c_s_avg.n[n_indices]
    end
    # Return the residuals
    c_s_star = [c_s_star_p; c_s_star_n]

    states[:c_s_star] = state_new(c_s_star, (:p, :n), p)

    return nothing
end

function build_OCV!(states, p::AbstractParam)
    """
    Calculate the open circuit voltages for the positive & negative electrodes
    """
    c_s_star = states[:c_s_star]
    T = states[:T]
    
    # Put the surface concentration into a fraction
    θ_p = c_s_star.p./p.θ[:c_max_p]
    θ_n = c_s_star.n./p.θ[:c_max_n]
    
    # Compute the OCV for the positive & negative electrodes.
    U_p, ∂U∂T_p = p.numerics.OCV_p(θ_p, T.p, p)
    U_n, ∂U∂T_n = p.numerics.OCV_n(θ_n, T.n, p)

    states[:U] = state_new([U_p; U_n], (:p, :n), p)
    states[:∂U∂T] = state_new([∂U∂T_p; ∂U∂T_n], (:p, :n), p)

    return nothing
end

function build_η!(states, p::AbstractParam)
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

function build_K_eff!(states, p::AbstractParam)
    c_e = states[:c_e]
    T = states[:T]

    K_eff_p, K_eff_s, K_eff_n = p.numerics.K_eff(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)

    states[:K_eff] = state_new([K_eff_p; K_eff_s; K_eff_n], (:p, :s, :n), p)
    
    return nothing
end

function build_heat_generation_rates!(states, p::AbstractParam)
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

    c_e_p = c_e.p
    c_e_s = c_e.s
    c_e_n = c_e.n

    T_p = T.p
    T_s = T.s
    T_n = T.n

    a_p, a_n = surface_area_to_volume_ratio(p)
    σ_eff_p, σ_eff_n = conductivity_effective(p)

    function thermal_derivatives(Φ_s, Φ_e, c_e, p)

        # For each of the numerical derivatives computed below; the first & last control volumes are evaluated with first
        # order accuracy [forward & backward difference schemes respectively]
        # while the middle control volume approximations use a second order accuracy [central difference scheme].
    
        Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)
    
        ## Solid potential derivatives
    
        # Positive Electrode
        dΦ_sp = [(-3*Φ_s[1]+4*Φ_s[2]-Φ_s[3])/(2*Δx_p*p.θ[:l_p]);           					# Forward differentiation scheme
            (Φ_s[3:p.N.p]-Φ_s[1:p.N.p-2]) / (2*Δx_p*p.θ[:l_p]);						# Central differentiation scheme
            (3*Φ_s[p.N.p]-4*Φ_s[p.N.p-1]+Φ_s[p.N.p-2]) / (2*Δx_p*p.θ[:l_p])		# Backward differentiation scheme
            ]
    
        # Negative Electrode
        dΦ_sn = [(-3*Φ_s[p.N.p+1]+4*Φ_s[p.N.p+2]-Φ_s[p.N.p+3])/(2*Δx_n*p.θ[:l_n]); 	# Forward differentiation scheme
            (Φ_s[p.N.p+3:end]-Φ_s[p.N.p+1:end-2]) / (2*Δx_n*p.θ[:l_n]); 					# Central differentiation scheme
            (3*Φ_s[end]-4*Φ_s[end-1]+Φ_s[end-2]) / (2*Δx_n*p.θ[:l_n]) 						# Backward differentiation scheme
            ]
    
        dΦ_s = [
            dΦ_sp
            dΦ_sn
            ]
    
        ## Electrolyte potential derivatives
    
        # Positive Electrode
    
        dΦ_ep = [ (-3*Φ_e[1]+4*Φ_e[2]-Φ_e[3])/(2*Δx_p*p.θ[:l_p]);		# Forward differentiation scheme
            (Φ_e[3:p.N.p]-Φ_e[1:p.N.p-2])/(2*Δx_p*p.θ[:l_p])	  	# Central differentiation scheme
            ]
    
        # Attention! The last volume of the positive electrode will involve one volume of the
        # separator for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # Last CV in the positive electrode: derivative approximation with a central scheme
        dΦ_e_last_p = 2*(Φ_e[p.N.p+1]-Φ_e[p.N.p-1])/(3 * Δx_p*p.θ[:l_p] + Δx_s*p.θ[:l_s])
    
        # Separator
    
        # Attention! The first volume of the separator will involve one volume of the
        # positive section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # First CV in the separator: derivative approximation with a central difference scheme
        dΦ_e_first_s = 2*(Φ_e[p.N.p+2]-Φ_e[p.N.p])/(Δx_p*p.θ[:l_p] + 3* Δx_s*p.θ[:l_s])
    
        # Central difference scheme
        dΦ_es =  (Φ_e[p.N.p+3:p.N.p+p.N.s]-Φ_e[p.N.p+1:p.N.p+p.N.s-2])/(2*Δx_s*p.θ[:l_s])
    
        # Attention! The last volume of the separator will involve one volume of the
        # negative section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # Last CV in the separator: derivative approximation with a central scheme
        dΦ_e_last_s = 2*(Φ_e[p.N.p+p.N.s+1]-Φ_e[p.N.p+p.N.s-1])/( Δx_n*p.θ[:l_n] + 3*Δx_s*p.θ[:l_s])
    
        # Negative electrode
    
        # Attention! The first volume of the negative electrode will involve one volume of the
        # separator section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # First CV in the negative electrode: derivative approximation with a central scheme
        dΦ_e_first_n = 2*(Φ_e[p.N.p+p.N.s+2]-Φ_e[p.N.p+p.N.s])/(3 * Δx_n*p.θ[:l_n] + Δx_s*p.θ[:l_s])
    
        # Central difference scheme
        dΦ_en = [(Φ_e[p.N.p+p.N.s+3:end]-Φ_e[p.N.p+p.N.s+1:end-2])/(2*Δx_n*p.θ[:l_n]);
            (3*Φ_e[end]-4*Φ_e[end-1]+Φ_e[end-2])/(2*Δx_n*p.θ[:l_n])
            ]
        dΦ_e = [
            dΦ_ep
            dΦ_e_last_p
            dΦ_e_first_s
            dΦ_es
            dΦ_e_last_s
            dΦ_e_first_n
            dΦ_en
        ]
    
        ## Electrolyte concentration derivatives
    
        # Positive Electrode
    
        dc_ep = [ (-3*c_e[1]+4*c_e[2]-c_e[3])/(2*Δx_p*p.θ[:l_p]); 		# Forward differentiation scheme
            (c_e[3:p.N.p]-c_e[1:p.N.p-2])/(2*Δx_p*p.θ[:l_p]) 	# Central differentiation scheme
            ]
    
        # Attention! The last volume of the positive electrode will involve one volume of the
        # separator for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # Last CV in the positive electrode: derivative approximation with a central scheme
        dc_e_last_p = 2*(c_e[p.N.p+1]-c_e[p.N.p-1])/(3 * Δx_p*p.θ[:l_p] + Δx_s*p.θ[:l_s])
    
        # Separator
    
        # Attention! The first volume of the separator will involve one volume of the
        # positive section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # First CV in the separator: derivative approximation with a central scheme
        dc_e_first_s = 2*(c_e[p.N.p+2]-c_e[p.N.p])/( Δx_p*p.θ[:l_p] + 3* Δx_s*p.θ[:l_s])
    
        # Central differentiation scheme
        dc_es = (c_e[p.N.p+3:p.N.p+p.N.s]-c_e[p.N.p+1:p.N.p+p.N.s-2])/(2*Δx_s*p.θ[:l_s])
    
        # Attention! The last volume of the separator will involve one volume of the
        # negative section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # Last CV in the separator: derivative approximation with a central scheme
        dc_e_last_s = 2*(c_e[p.N.p+p.N.s+1]-c_e[p.N.p+p.N.s-1])/( Δx_n*p.θ[:l_n] + 3*Δx_s*p.θ[:l_s])
    
        # Negative electrode
    
        # Attention! The first volume of the negative electrode will involve one volume of the
        # separator section for the calculation of the derivative. Therefore suitable
        # considerations must be done with respect to the deltax quantities.
    
        # First CV in the negative electrode: derivative approximation with a central scheme
        dc_e_first_n = 2*(c_e[p.N.p+p.N.s+2]-c_e[p.N.p+p.N.s])/(3 * Δx_n*p.θ[:l_n] + Δx_s*p.θ[:l_s])
    
        dc_en = [(c_e[p.N.p+p.N.s+3:end]-c_e[p.N.p+p.N.s+1:end-2])/(2*Δx_p*p.θ[:l_p]); 	# Central differentiation scheme
            (3*c_e[end]-4*c_e[end-1]+c_e[end-2])/(2*Δx_n*p.θ[:l_n]) 						# Backward differentiation scheme
            ]
    
        dc_e = [
            dc_ep
            dc_e_last_p
            dc_e_first_s
            dc_es
            dc_e_last_s
            dc_e_first_n
            dc_en
        ]
    
        return dΦ_s, dΦ_e, dc_e
    end

    # Evaluate the derivatives used in Q_ohm calculations
    dΦ_s, dΦ_e, dc_e = thermal_derivatives(Φ_s, Φ_e, c_e, p)

    ## Reversible heat generation rate

    # Positive electrode
    @views @inbounds Q_rev_p = F*a_p*j[1:p.N.p].*T[p.N.a+1:p.N.a+p.N.p].*∂U∂T.p

    # Negative Electrode
    @views @inbounds Q_rev_n = F*a_n*j[p.N.p+1:end].*T[p.N.a+p.N.p+p.N.s+1:p.N.a+p.N.p+p.N.s+p.N.n].*∂U∂T.n

    ## Reaction heat generation rate

    # Positive overpotential
    @views @inbounds η_p = @. (Φ_s[1:p.N.p]-Φ_e[1:p.N.p]-U.p)
    # Positive reaction heat generation rate
    @views @inbounds Q_rxn_p = F*a_p*j[1:p.N.p].*η.p

    # Negative overpotential
    @views @inbounds η_n = @. (Φ_s[p.N.p+1:end]-Φ_e[p.N.p+p.N.s+1:end]-U.n)
    # Negative reaction heat generation rate
    @views @inbounds Q_rxn_n = F*a_n*j[p.N.p+1:end].*η.n

    ## Ohmic heat generation rate
    # Positive electrode ohmic generation rate
    @views @inbounds Q_ohm_p = σ_eff_p * (dΦ_s[1:p.N.p]).^2 + K_eff.p.*(dΦ_e[1:p.N.p]).^2 + 2*R*K_eff.p.*T[p.N.a+1:p.N.a+p.N.p]*(1-p.θ[:t₊])/F.*dc_e[1:p.N.p].*1.0./c_e[1:p.N.p].*dΦ_e[1:p.N.p]
    # Separator ohmic generation rate
    @views @inbounds Q_ohm_s = K_eff.s.*(dΦ_e[p.N.p+1:p.N.p+p.N.s]).^2 + 2*R*K_eff.s.*T[p.N.a+p.N.p+1:p.N.a+p.N.p+p.N.s]*(1-p.θ[:t₊])/F.*dc_e[p.N.p+1:p.N.p+p.N.s].*1.0./c_e[p.N.p+1:p.N.p+p.N.s].*dΦ_e[p.N.p+1:p.N.p+p.N.s]
    # Negative electrode ohmic generation rate
    Q_ohm_n = σ_eff_n * (dΦ_s[p.N.p+1:end]).^2 +K_eff.n.*(dΦ_e[p.N.p+p.N.s+1:end]).^2 + 2*R*K_eff.n.*T[p.N.a+p.N.p+p.N.s+1:p.N.a+p.N.p+p.N.s+p.N.n]*(1-p.θ[:t₊])/F.*dc_e[p.N.p+p.N.s+1:end].*1.0./c_e[p.N.p+p.N.s+1:end].*dΦ_e[p.N.p+p.N.s+1:end]
    
    Q_rev = [Q_rev_p; Q_rev_n]
    Q_rxn = [Q_rxn_p; Q_rxn_n]
    Q_ohm = [Q_ohm_p; Q_ohm_s; Q_ohm_n]

    states[:Q_rev] = state_new(Q_rev, (:p, :n), p)
    states[:Q_rxn] = state_new(Q_rxn, (:p, :n), p)
    states[:Q_ohm] = state_new(Q_ohm, (:p, :s, :n), p)

    return nothing
end

function build_residuals!(res_tot::AbstractVector, res::Dict, p::AbstractParam)
    """
    Create the residuals vector using all the variables which are needed in the simulation.
    `p.cache.vars` contains the list of variables, and `getproperty(p.ind, var)` will
    retrieve the appropriate indices for all the variables
    """
    @inbounds for var in p.cache.vars
        ind_var = getproperty(p.ind, var)
        res_tot[ind_var] .= res[var]
    end
    return nothing
end


"""
Constants and coefficients
"""
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


"""
Calculations which are primarily used in `set_vars!`. Denoted by the prefix `calc_`.
Since p.θ is a dictionary which may contain `Float64` or `Vector{Float64}`, it is important
to denote the type of each variable for performance
"""
@inline function calc_I1C(p::param)
    """
    Calculate the 1C current density (A/m²)
    """
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
    """
    Calculate the voltage (V)
    """
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
    """
    Calculate the current (C-rate)
    """
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
    """
    Calculate the power (W)
    """
    if run.method === :I || run.method === :V
        P = @views @inbounds model.I[end]*model.V[end]
    elseif run.method === :P
        P = value(run)
    end

    return P
end

@inline function calc_SOC(c_s_avg::AbstractVector{Float64}, p::param)
    """
    Calculate the SOC (dimensionless fraction)
    """
    if p.numerics.solid_diffusion === :Fickian
        c_s_avg_sum = @views @inbounds mean(c_s_avg[(p.N.p*p.N.r_p)+1:end])
    else # c_s_avg in neg electrode
        c_s_avg_sum = @views @inbounds mean(c_s_avg[p.N.p+1:end])
    end

    return (c_s_avg_sum/p.θ[:c_max_n]::Float64 - p.θ[:θ_min_n]::Float64)/(p.θ[:θ_max_n]::Float64 - p.θ[:θ_min_n]::Float64) # cell-soc fraction
end
