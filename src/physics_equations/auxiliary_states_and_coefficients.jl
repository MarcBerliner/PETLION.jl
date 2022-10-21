"""
Much of the code in this file is adapted from LIONSIMBA, a li-ion simulation toolbox in MATLAB.
See https://github.com/lionsimbatoolbox/LIONSIMBA for more information.
"""

function build_auxiliary_states!(states, p::AbstractModel)
    """
    Calculates necessary auxiliary states and adds them to the `state` dict
    """
    
    # create a state j_total which is a combination of j and j_s
    build_j_total!(states, p)

    # The final value of the x vector can either be I or P,
    # so this function specifically designates the I and P states
    build_I_V!(states, p)

    # Active material, ϵ_s
    build_ϵ_s!(states, p)

    # Porisity, ϵ
    build_ϵ!(states, p)

    # Conductivity effective, σ_eff
    build_σ_eff_p!(states, p)

    # Surface area to volume ratio, a
    build_a!(states, p)

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
    
    # Electrolyte diffusion, D_eff
    build_D_eff!(states, p)

    # Solid diffusion, D_s_eff
    build_D_s_eff!(states, p)
    
    return nothing
end

function build_I_V!(states, p::AbstractModel)
    """
    Define the current and voltage
    """

    Φ_s = states[:Φ_s]

    I1C = calc_I1C(p)
    
    I = states[:I][1]*I1C
    V = Φ_s[1] - Φ_s[end]
    
    states[:I] = state_new([I], (), p)
    states[:V] = state_new([V], (), p)

    return nothing
end

function build_ϵ_s!(states, p::AbstractModel)
    """
    If active material is not a state, include a vector of active material fractions for each section [-]
    """
    ϵ_sp, ϵ_sn = active_material(p)
    ϵ_sp = ϵ_sp*ones(p.N.p)
    ϵ_sn = ϵ_sn*ones(p.N.n)

    if haskey(states, :ϵ_s)
        ϵ_sp = iszero(p.ind.ϵ_s.p) ? ϵ_sp : states[:ϵ_s].p
        ϵ_sn = iszero(p.ind.ϵ_s.n) ? ϵ_sn : states[:ϵ_s].n
    end
    
    ϵ_s = [ϵ_sp;ϵ_sn]

    states[:ϵ_s] = state_new(ϵ_s, (:p, :n), p)

    return nothing
end

function build_ϵ!(states, p::AbstractModel)
    """
    Electrode porosity fraction [-]
    """
    ϵ_s = states[:ϵ_s]

    ϵ_p_porosity = 1.0 .- (p.θ[:ϵ_fp] .+ ϵ_s.p)
    ϵ_s_porosity = p.θ[:ϵ_s]*ones(p.N.s)
    ϵ_n_porosity = 1.0 .- (p.θ[:ϵ_fn] .+ ϵ_s.n)

    ϵ = [ϵ_p_porosity; ϵ_s_porosity; ϵ_n_porosity]

    states[:ϵ] = state_new(ϵ, (:p, :s, :n), p)
end

function build_σ_eff_p!(states, p::AbstractModel)
    """
    Effective conductivity [S/m]
    """
    ϵ_s = states[:ϵ_s]

    σ_eff_p = p.θ[:σ_p]*ϵ_s.p
    σ_eff_n = p.θ[:σ_n]*ϵ_s.n

    σ_eff = [σ_eff_p; σ_eff_n]

    states[:σ_eff] = state_new(σ_eff, (:p, :n), p)

    return nothing
end

function build_a!(states, p::AbstractModel)
    """
    Surface area to volume ratio for a sphere (SA/V = 4πr^2/(4/3πr^3)) multipled by the active material fraction [m^2/m^3]
    """
    ϵ_s = states[:ϵ_s]

    a_p = 3ϵ_s.p/p.θ[:Rp_p]
    a_n = 3ϵ_s.n/p.θ[:Rp_n]

    a = [a_p; a_n]

    states[:a] = state_new(a, (:p, :n), p)

    return nothing
end

function substitute_porosities!(X, states, sections::Tuple, p::AbstractModel)
    """
    Some effective functions are written with a constant porosity, but
    this function replaces the constant porosity with the state porosity
    """
    ϵ = states[:ϵ]
    
    start = 0
    for section in sections
        N = getfield(p.N, section)
        name = Symbol("ϵ_", section)

        for i in (1:N) .+ start
            X[i] = substitute(X[i], p.θ[name] => ϵ[i])
        end
        start += N
    end

    return X
end

function build_j_total!(states, p::AbstractModel)
    """
    Append `j` with possible additions from `j_s`
    """
    j = states[:j]
    j_s = states[:j_s]
    
    if !is_active(j_s)
        states[:j_total] = state_new(j, (:p, :n), p)
        return nothing
    end

    j_total = copy(j)
    j_total[p.N.p+1:end] .+= j_s

    states[:j_total] = state_new(j_total, (:p, :n), p)

    return nothing
end

build_T!(states, p::AbstractModelTemp{true}) = nothing
function build_T!(states, p::AbstractModelTemp{false})
    """
    If temperature is not enabled, include a vector of temperatures using the specified initial temperature.
    """
    T = repeat([p.θ[:T₀]], (p.N.p+p.N.s+p.N.n+p.N.a+p.N.z))
    
    states[:T] = state_new(T, (:a, :p, :s, :n, :z), p)
    
    return nothing
end


function build_c_s_star!(states, p::AbstractModelSolidDiff{:Fickian})
    """
    Evaluates the concentration of Li-ions at the electrode surfaces.
    """
    c_s_avg = states[:c_s_avg]
    
    p_indices = c_s_indices(p,:p; surf=true,offset=false)
    n_indices = c_s_indices(p,:n; surf=true,offset=false)
    
    c_s_star_p = c_s_avg[p_indices]
    c_s_star_n = c_s_avg[n_indices]
    
    # Return the residuals
    c_s_star = [c_s_star_p; c_s_star_n]
    
    states[:c_s_star] = state_new(c_s_star, (:p, :n), p)
    
    return nothing
end
function build_c_s_star!(states, p::AbstractModelSolidDiff{:quadratic})
    """
    Evaluates the concentration of Li-ions at the electrode surfaces.
    """
    c_s_avg = states[:c_s_avg]
    j = states[:j]
    D_s_eff = states[:D_s_eff]
    
    # Evaluates the average surface concentration in both the electrodes.
    c_s_star_p = c_s_avg.p-(p.θ[:Rp_p]./(D_s_eff.p.*5)).*j.p
    c_s_star_n = c_s_avg.n-(p.θ[:Rp_n]./(D_s_eff.n.*5)).*j.n
    
    # Return the residuals
    c_s_star = [c_s_star_p; c_s_star_n]
    
    states[:c_s_star] = state_new(c_s_star, (:p, :n), p)
    
    return nothing
end
function build_c_s_star!(states, p::AbstractModelSolidDiff{:polynomial})
    """
    Evaluates the concentration of Li-ions at the electrode surfaces.
    """
    c_s_avg = states[:c_s_avg]
    j = states[:j]
    Q = states[:Q]
    D_s_eff = states[:D_s_eff]
    
    c_s_star_p = c_s_avg.p+(p.θ[:Rp_p]./(D_s_eff.p.*35)).*(-j.p+8*D_s_eff.p.*Q.p)
    c_s_star_n = c_s_avg.n+(p.θ[:Rp_n]./(D_s_eff.n.*35)).*(-j.n+8*D_s_eff.n.*Q.n)
    
    # Return the residuals
    c_s_star = [c_s_star_p; c_s_star_n]

    states[:c_s_star] = state_new(c_s_star, (:p, :n), p)

    return nothing
end

function build_OCV!(states, p::AbstractModel)
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

function build_η!(states, p::AbstractModel)
    """
    Calculate the overpotentials for the positive & negative electrodes
    """

    Φ_s = states[:Φ_s]
    Φ_e = states[:Φ_e]
    U = states[:U]
    j = states[:j]
    film = states[:film]

    F = const_Faradays

    η_p = @. Φ_s.p - Φ_e.p - U.p
    η_n = @. Φ_s.n - Φ_e.n - U.n

    if haskey(p.θ, :R_film_n)
        η_n .+= -j.n.*F.*p.θ[:R_film_n]
    end
    
    if     p.numerics.aging == :SEI
        R_film = p.θ[:R_SEI] .+ film./p.θ[:k_n_aging]
        η_n .+= @. - F*j.n*R_film
    end

    states[:η] = state_new([η_p; η_n], (:p, :n), p)

    return nothing
end

function build_K_eff!(states, p::AbstractModel)
    c_e = states[:c_e]
    T = states[:T]

    K_eff_p, K_eff_s, K_eff_n = p.numerics.K_eff(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)

    K_eff = [K_eff_p; K_eff_s; K_eff_n]
    substitute_porosities!(K_eff, states, (:p, :s, :n), p)

    states[:K_eff] = state_new(K_eff, (:p, :s, :n), p)
    
    return nothing
end

function build_D_eff!(states, p::AbstractModel)
    c_e = states[:c_e]
    T = states[:T]

    D_eff_p, D_eff_s, D_eff_n = p.numerics.D_eff(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)

    D_eff = [D_eff_p; D_eff_s; D_eff_n]
    substitute_porosities!(D_eff, states, (:p, :s, :n), p)

    states[:D_eff] = state_new(D_eff, (:p, :s, :n), p)
    
    return nothing
end

function build_D_s_eff!(states, p::AbstractModel)
    c_s_avg = states[:c_s_star]
    T = states[:T]

    D_s_eff_p, D_s_eff_n = p.numerics.D_s_eff(c_s_avg.p, c_s_avg.n, T.p, T.n, p)

    D_s_eff = [D_s_eff_p; D_s_eff_n]
    substitute_porosities!(D_s_eff, states, (:p, :n), p)

    states[:D_s_eff] = state_new(D_s_eff, (:p, :n), p)
    
    return nothing
end

function build_heat_generation_rates!(states, p::AbstractModel)
    """
    Evaluates the heat source terms used in the thermal sol per section
    """

    Φ_s = states[:Φ_s]
    Φ_e = states[:Φ_e]
    j = states[:j_total]
    T = states[:T]
    c_e = states[:c_e]
    ∂U∂T = states[:∂U∂T]
    η = states[:η]
    K_eff = states[:K_eff]
    a = states[:a]
    σ_eff = states[:σ_eff]

    F = const_Faradays
    R = const_Ideal_Gas

    function thermal_derivatives(Φ_s, Φ_e, c_e, p)
        """
        Computing approximations to ∂/∂x for Φ_s, Φ_e, and c_e
        """
    
        Δx = Δx_values(p.N)
    
        ## Solid potential derivatives

        function forward_difference_left(x, state::Symbol)
            Δx_c = getfield(Δx, state)
            l = p.θ[Symbol(:l_, state)]
            out = (-3x[1] + 4x[2] - x[3]) / (2Δx_c*l)

            return out
        end
        forward_difference_right(x,state;kw...) = -forward_difference_left(reverse(x),state;kw...)

        function central_difference(x, state::Symbol)
            Δx_c = getfield(Δx, state)
            l = p.θ[Symbol(:l_, state)]
            out = (x[3:end] - x[1:end-2]) / (2Δx_c*l)

            return out
        end

        function approx_central_difference_at_right_CV(x_l, state_l::Symbol, x_r, state_r::Symbol)
            Δx_l = getfield(Δx, state_l)
            l_l = p.θ[Symbol(:l_, state_l)]

            Δx_r = getfield(Δx, state_r)
            l_r = p.θ[Symbol(:l_, state_r)]
            
            out = 2(x_r[1] - x_l[end-1]) / (3Δx_l*l_l + Δx_r*l_r)

            return out
        end
        function approx_central_difference_at_left_CV(x_l, state_l::Symbol, x_r, state_r::Symbol)
            Δx_l = getfield(Δx, state_l)
            l_l = p.θ[Symbol(:l_, state_l)]

            Δx_r = getfield(Δx, state_r)
            l_r = p.θ[Symbol(:l_, state_r)]
            
            out = 2*(x_r[2] - x_l[end]) / (Δx_l*l_l + 3Δx_r*l_r)
        
            return out
        end
    
        # Cathode
        dΦ_sp = [
            forward_difference_left(Φ_s.p, :p)
            central_difference(Φ_s.p, :p)
            forward_difference_right(Φ_s.p, :p)
        ]
    
        # Anode
        dΦ_sn = [
            forward_difference_left(Φ_s.n, :n)
            central_difference(Φ_s.n, :n)
            forward_difference_right(Φ_s.n, :n)
        ]
    
        dΦ_s = (
            p = dΦ_sp,
            n = dΦ_sn,
        )
    
        ## Electrolyte potential derivatives
    
        # Cathode
    
        dΦ_ep = [
            forward_difference_left(Φ_e.p, :p)
            central_difference(Φ_e.p, :p)
            approx_central_difference_at_right_CV(Φ_e.p, :p, Φ_e.s, :s)
        ]
    
        dΦ_es = [
            approx_central_difference_at_left_CV(Φ_e.p, :p, Φ_e.s, :s)
            central_difference(Φ_e.s, :s)
            approx_central_difference_at_right_CV(Φ_e.s, :s, Φ_e.n, :n)
        ]
        
        dΦ_en = [
            approx_central_difference_at_left_CV(Φ_e.s, :s, Φ_e.n, :n)
            central_difference(Φ_e.n, :n)
            forward_difference_right(Φ_e.n, :n)
        ]
        
        dΦ_e = (
            p = dΦ_ep,
            s = dΦ_es,
            n = dΦ_en,
        )
    
        ## Electrolyte concentration derivatives
    
        dc_ep = [
            forward_difference_left(c_e.p, :p)
            central_difference(c_e.p, :p)
            approx_central_difference_at_right_CV(c_e.p, :p, c_e.s, :s)
        ]

        dc_es = [
            approx_central_difference_at_left_CV(c_e.p, :p, c_e.s, :s)
            central_difference(c_e.s, :s)
            approx_central_difference_at_right_CV(c_e.s, :s, c_e.n, :n)
        ]

        dc_en = [
            approx_central_difference_at_left_CV(c_e.s, :s, c_e.n, :n)
            central_difference(c_e.n, :n)
            forward_difference_right(c_e.n, :n)
        ]

        dc_e = (
            p = dc_ep,
            s = dc_es,
            n = dc_en,
        )
    
        return dΦ_s, dΦ_e, dc_e
    end

    # Evaluate the derivatives used in Q_ohm calculations
    dΦ_s, dΦ_e, dc_e = thermal_derivatives(Φ_s, Φ_e, c_e, p)

    ## Reversible heat generation rate
    @views @inbounds Q_rev_p = F.*a.p.*j.p.*T.p.*∂U∂T.p
    @views @inbounds Q_rev_n = F.*a.n.*j.n.*T.n.*∂U∂T.n

    ## Reaction heat generation rate
    @views @inbounds Q_rxn_p = F.*a.p.*j.p.*η.p
    @views @inbounds Q_rxn_n = F.*a.n.*j.n.*η.n

    ## Ohmic heat generation rate
    ν_p, ν_s, ν_n = p.numerics.thermodynamic_factor(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)

    # Cathode ohmic generation rate
    Q_ohm_p = K_eff.p .* dΦ_e.p.^2 + 2*R*K_eff.p.*T.p*(1-p.θ[:t₊]).*ν_p/F.*(dc_e.p./c_e.p).*dΦ_e.p .+ σ_eff.p .* dΦ_s.p.^2
    # Separator ohmic generation rate
    Q_ohm_s = K_eff.s .* dΦ_e.s.^2 + 2*R*K_eff.s.*T.s*(1-p.θ[:t₊]).*ν_s/F.*(dc_e.s./c_e.s).*dΦ_e.s
    # Anode ohmic generation rate
    Q_ohm_n = K_eff.n .* dΦ_e.n.^2 + 2*R*K_eff.n.*T.n*(1-p.θ[:t₊]).*ν_n/F.*(dc_e.n./c_e.n).*dΦ_e.n .+ σ_eff.n .* dΦ_s.n.^2
    
    Q_rev = [Q_rev_p; Q_rev_n]
    Q_rxn = [Q_rxn_p; Q_rxn_n]
    Q_ohm = [Q_ohm_p; Q_ohm_s; Q_ohm_n]

    states[:Q_rev] = state_new(Q_rev, (:p, :n), p)
    states[:Q_rxn] = state_new(Q_rxn, (:p, :n), p)
    states[:Q_ohm] = state_new(Q_ohm, (:p, :s, :n), p)

    return nothing
end

function build_residuals!(res_tot::AbstractVector, res::Dict, p::AbstractModel)
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
function active_material(p::AbstractModel)
    """
    Electrode active material fraction [-]
    """
    ϵ_sp = 1.0 - (p.θ[:ϵ_fp] + p.θ[:ϵ_p])
    ϵ_sn = 1.0 - (p.θ[:ϵ_fn] + p.θ[:ϵ_n])

    return ϵ_sp, ϵ_sn
end

function conductivity_effective(p::AbstractModel)
    """
    Effective conductivity [S/m]
    """
    ϵ_sp, ϵ_sn = active_material(p)

    σ_eff_p = p.θ[:σ_p]*ϵ_sp
    σ_eff_n = p.θ[:σ_n]*ϵ_sn

    return σ_eff_p, σ_eff_n
end

function surface_area_to_volume_ratio(p::AbstractModel)
    """
    Surface area to volume ratio for a sphere (SA/V = 4πr^2/(4/3πr^3)) multipled by the active material fraction [m^2/m^3]
    """
    ϵ_sp, ϵ_sn = active_material(p)

    a_p = 3ϵ_sp/p.θ[:Rp_p]
    a_n = 3ϵ_sn/p.θ[:Rp_n]

    return a_p, a_n
end

function coeff_reaction_rate(states, p::AbstractModel)
    """
    Reaction rates (k) of cathode and anode [m^2.5/(m^0.5 s)]
    """
    T = states[:T]
    c_s_avg = states[:c_s_avg]

    return p.numerics.rxn_rate(T.p, T.n, c_s_avg.p, c_s_avg.n, p)
end

function coeff_solid_diffusion_effective(states::Dict, p::AbstractModel)
    c_s_avg = states[:c_s_avg]
    T = states[:T]
    
    return p.numerics.D_s_eff(c_s_avg.p, c_s_avg.n, T.p, T.n, p)
end

function coeff_electrolyte_diffusion_effective(states::Dict, p::AbstractModel)
    c_e = states[:c_e]
    T = states[:T]
    
    return p.numerics.D_eff(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)
end

function calc_j_analytic(Y::AbstractVector, p::AbstractModel)
    T_p = repeat([p.θ[:T₀]],p.N.p)
    T_n = repeat([p.θ[:T₀]],p.N.n)

    c_s_avg_p = Y[p.ind.c_s_avg.p]
    c_s_avg_n = Y[p.ind.c_s_avg.n]

    c_s_star_p = Y[p.ind.c_s_avg.p[p.N.r_p:p.N.r_p:p.N.r_p*p.N.p]]
    c_s_star_n = Y[p.ind.c_s_avg.n[p.N.r_n:p.N.r_n:p.N.r_n*p.N.n]]

    # Calculate the reaction rates
    k_p_eff, k_n_eff = p.numerics.rxn_rate(T_p, T_n, c_s_avg_p, c_s_avg_n, p)
    
    j_p_calc = p.numerics.rxn_p(c_s_star_p, Y[p.ind.c_e.p], T_p, Y[p.ind.Φ_s.p] .- Y[p.ind.Φ_e.p] .- p.numerics.OCV_p(c_s_star_p./p.θ[:c_max_p],T_p,p)[1], k_p_eff, p.θ[:λ_MHC_p], p.θ[:c_max_p], p)
    j_n_calc = p.numerics.rxn_n(c_s_star_n, Y[p.ind.c_e.n], T_n, Y[p.ind.Φ_s.n] .- Y[p.ind.Φ_e.n] .- p.numerics.OCV_n(c_s_star_n./p.θ[:c_max_n],T_n,p)[1], k_n_eff, p.θ[:λ_MHC_n], p.θ[:c_max_n], p)

    return [j_p_calc; j_n_calc]
end

"""
Calculations which are primarily used in `set_vars!`, denoted by the prefix `calc_`.
"""
function limiting_electrode(p::AbstractModel)
    θ = p.θ
    ϵ_sp, ϵ_sn = active_material(p)

    Q_p = ϵ_sp*θ[:l_p]*θ[:c_max_p]*(θ[:θ_min_p] - θ[:θ_max_p])
    Q_n = ϵ_sn*θ[:l_n]*θ[:c_max_n]*(θ[:θ_max_n] - θ[:θ_min_n])

    if Q_p > Q_n
        return "anode", Q_n*PETLION.const_Faradays/3600.0
    else
        return "cathode", Q_p*PETLION.const_Faradays/3600.0
    end
end

@inline calc_I1C(p::AbstractModel) = calc_I1C(p.θ)
@inline function calc_I1C(θ::OrderedDict{Symbol,T}) where T<:Union{Float64,Any}
    """
    Calculate the 1C current density (A⋅hr/m²) based on the limiting electrode
    """
    F = const_Faradays
    
    ϵ_sp = 1.0 - (θ[:ϵ_fp] + θ[:ϵ_p])
    ϵ_sn = 1.0 - (θ[:ϵ_fn] + θ[:ϵ_n])

    I1C = (F/3600.0)*min(
        ϵ_sp*θ[:l_p]*θ[:c_max_p]*(θ[:θ_min_p] - θ[:θ_max_p]),
        ϵ_sn*θ[:l_n]*θ[:c_max_n]*(θ[:θ_max_n] - θ[:θ_min_n]),
        )

    return I1C
end

@inline function temperature_weighting(T::AbstractVector{<:Number},p::AbstractModel)
    l_a,l_p,l_s,l_n,l_z = (p.θ[:l_a], p.θ[:l_p], p.θ[:l_s], p.θ[:l_n], p.θ[:l_z])

    ratio_a = l_a/p.N.a
    ratio_p = l_p/p.N.p
    ratio_s = l_s/p.N.s
    ratio_n = l_n/p.N.n
    ratio_z = l_z/p.N.z

    T_mean = 0.0
    @inbounds for i in (1:p.N.a)
        T_mean += T[i]*ratio_a
    end
    @inbounds for i in (1:p.N.p) .+ (p.N.a)
        T_mean += T[i]*ratio_p
    end
    @inbounds for i in (1:p.N.s) .+ (p.N.a+p.N.p)
        T_mean += T[i]*ratio_s
    end
    @inbounds for i in (1:p.N.n) .+ (p.N.a+p.N.p+p.N.s)
        T_mean += T[i]*ratio_n
    end
    @inbounds for i in (1:p.N.z) .+ (p.N.a+p.N.p+p.N.s+p.N.n)
        T_mean += T[i]*ratio_z
    end

    return T_mean/(l_a+l_p+l_s+l_n+l_z)
end
@inline function constant_temperature(t,Y,YP::AbstractVector{<:Number},p::AbstractModel)
    temperature_weighting((@views @inbounds YP[p.ind.T]),p)
end
temperature_weighting(T::VectorOfArray,p::AbstractModel) = [temperature_weighting(_T,p) for _T in T]

state_deriv_func(::Val{index}) where {index} = (t,Y,YP::AbstractVector{<:Number},p::AbstractModel) -> YP[index]
state_deriv_func(index::Int64) = state_deriv_func(Val(index))

state_func(::Val{index}) where {index} = (t,Y::AbstractVector{<:Number},YP,p::AbstractModel) -> Y[index]
state_func(index::Int64) = state_func(Val(index))

function c_s_indices(p::AbstractModelSolidDiff{:Fickian}, section::Symbol; surf::Bool=true,offset::Bool=true)
    N = p.N

    ind_p = N.r_p:N.r_p:N.r_p*N.p
    ind_n = N.r_n:N.r_n:N.r_n*N.n
    
    return _c_s_indices(p, ind_p, ind_n, section, surf, offset)
end
function c_s_indices(p::Union{AbstractModelSolidDiff{:quadratic},AbstractModelSolidDiff{:polynomial}}, section::Symbol; surf::Bool=true,offset::Bool=true)
    N = p.N

    ind_p = 1:N.p
    ind_n = 1:N.n
    
    return _c_s_indices(p, ind_p, ind_n, section, surf, offset)
end
function _c_s_indices(p::AbstractModel,ind_p::T,ind_n::T,section::Symbol,surf::Bool,offset::Bool) where T<:AbstractRange{Int64}
    ind = p.ind.c_s_avg

    if     section == :p
        ind_final = surf ? ind.p[ind_p] : ind.p
    elseif section == :n
        ind_final = surf ? ind.n[ind_n] : ind.n
    else
        error("Section must be either `:p` or `:n`.")
    end

    if !offset
        ind_final = ind_final .- (ind.start-1)
    end

    return ind_final
end
