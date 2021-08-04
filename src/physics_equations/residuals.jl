function residuals_PET!(residuals, t, x, ẋ, p::AbstractParam)
    
    """
    First put the vector of x's, ẋ's, and residuals into dictionaries
    """
    states = retrieve_states(x, p)
    ∂states = retrieve_states(ẋ, p)
    res = retrieve_states(residuals, p)

    states[:t] = t
    states[:x] = x
    states[:ẋ] = ẋ

    """
    Calculate a few necessary auxiliary variables
    """
    build_auxiliary_states!(states, p)
    
    """
    Differential residuals
    """
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
    if p.numerics.temperature === true
        build_heat_generation_rates!(states, p)
        residuals_T!(res, states, ∂states, p)
    end

    """
    Algebraic residuals
    """
    # Residuals for ionic flux, j
    residuals_j!(res, states, p)
    
    # Residuals for side reaction ionic flux, j_s
    if p.numerics.aging ∈ (:SEI, :R_aging)
        residuals_j_s!(res, states, p)
    end

    # Residuals for the electrolyte potential, Φ_e
    residuals_Φ_e!(res, states, p)

    # Residuals for the solid potential, Φ_s
    residuals_Φ_s!(res, states, p)

    # Residuals for applied current density, I
    residuals_scalar!(res, states, p)

    """
    Compile all residuals together
    """
    build_residuals!(residuals, res, p)

    return nothing
end

function residuals_c_e!(res, states, ∂states, p::AbstractParam)
    """
    Calculate the electrolyte concentration residuals
    """

    c_e = states[:c_e]
    T   = states[:T]
    j   = states[:j_aging]

    ∂c_e = ∂states[:c_e]

    res_c_e = res[:c_e]

    Δx = Δx_values(p.N)

    # Diffusion coefficients
    # Comment this for benchmark purposes
    D_eff_p, D_eff_s, D_eff_n = coeff_electrolyte_diffusion_effective(states, p)
    a_p, a_n = surface_area_to_volume_ratio(p)

    # Interpolation of the diffusion coefficients, same for electrolyte conductivities
    D_eff_p, D_eff_s, D_eff_n = interpolate_electrolyte_conductivities(D_eff_p, D_eff_s, D_eff_n, p)

    A_tot = block_matrix_maker(p, -D_eff_p, -D_eff_s, -D_eff_n)

    # dividing by the length and Δx
    @views @inbounds A_tot[(1:p.N.p), (1:p.N.p)] ./= (Δx.p*p.θ[:l_p])^2
    @views @inbounds A_tot[(1:p.N.s) .+ (p.N.p), (1:p.N.s) .+ (p.N.p)] ./= (Δx.s*p.θ[:l_s])^2
    @views @inbounds A_tot[(1:p.N.n) .+ (p.N.p + p.N.s), (1:p.N.n) .+ (p.N.p + p.N.s)] ./= (Δx.n*p.θ[:l_n])^2

    # Reset values on the lines for the interfaces conditions
    @views @inbounds A_tot[p.N.p,:]         .= 0.0
    @views @inbounds A_tot[p.N.p+1,:]       .= 0.0

    # Reset values on the lines for the interfaces conditions
    @views @inbounds A_tot[p.N.p+p.N.s,:]   .= 0.0
    @views @inbounds A_tot[p.N.p+p.N.s+1,:] .= 0.0

    ## Interface between separator & positive electrode [last volume in the positive electrode]

    # Compute the common denominator at the interface
    @views @inbounds den_s = (Δx.p*p.θ[:l_p]/2 + Δx.s*p.θ[:l_s]/2)
    # Last diffusion coefficient of the positive electrode
    @views @inbounds last_p = D_eff_p[end-1]/(Δx.p*p.θ[:l_p])
    # Diffusion coefficient on the interface
    @views @inbounds first_s = D_eff_p[end]/den_s
    # Fix the values at the boundaries
    @views @inbounds A_tot[p.N.p,p.N.p-1:p.N.p+1] .= [last_p; -(last_p + first_s); first_s]/(Δx.p*p.θ[:l_p]*p.θ[:ϵ_p])

    ## Interface between separator & positive electrode [first volume in the separator]

    # First diffusion coefficient in the separator
    @views @inbounds second_s = D_eff_s[1]/(Δx.s*p.θ[:l_s])
    # Diffusion coefficient on the interface
    @views @inbounds first_s = D_eff_p[end]/den_s

    @views @inbounds A_tot[p.N.p+1,p.N.p:p.N.p+2] .= [first_s; -(first_s+second_s); second_s]/(Δx.s*p.θ[:l_s]*p.θ[:ϵ_s])

    ## Interface between separator & negative electrode [last volume in the separator]

    # Compute the common denominator at the interface
    @views @inbounds den_s = (Δx.s*p.θ[:l_s]/2 + Δx.n*p.θ[:l_n]/2)
    # Last diffusion coefficient in the separator
    @views @inbounds last_s = D_eff_s[end-1]/(Δx.s*p.θ[:l_s])
    # Diffusion coefficient on the interface
    @views @inbounds first_n = D_eff_s[end]/den_s

    @views @inbounds A_tot[p.N.p+p.N.s,p.N.p+p.N.s-1:p.N.p+p.N.s+1] = [last_s; -(last_s+first_n); first_n]/(Δx.s*p.θ[:l_s]*p.θ[:ϵ_s])

    ## Interface between separator & negative electrode [first volume in the negative electrode]

    # Compute the common denominator at the interface
    den_n = (Δx.s*p.θ[:l_s]/2 + Δx.n*p.θ[:l_n]/2)
    # First diffusion coefficient in the negative electrode
    @views @inbounds second_n = D_eff_n[1]/(Δx.n*p.θ[:l_n])
    # Diffusion coefficient on the interface
    @views @inbounds first_n = D_eff_s[end]/den_n

    
    A_tot[p.N.p+p.N.s+1,p.N.p+p.N.s:p.N.p+p.N.s+2] = [first_n; -(first_n+second_n); second_n]/(Δx.n*p.θ[:l_n]*p.θ[:ϵ_n])
    
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

    ν_p,ν_s,ν_n = p.numerics.thermodynamic_factor(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)

    rhsCe[ind_p] .+= K[ind_p].*(1-p.θ[:t₊]).*ν_p.*a_p.*j.p
    # nothing for the separator since a_s = 0
    rhsCe[ind_n] .+= K[ind_n].*(1-p.θ[:t₊]).*ν_n.*a_n.*j.n

    # Write the residual of the equation
    res_c_e .= rhsCe .- ∂c_e

    return nothing
end

function residuals_c_s_avg!(res, states, ∂states, p::AbstractParam)
    """
    Calculate the solid particle concentration residuals using various methods defined by p.numerics.solid_diffusion [mol/m³]
    """

    if p.numerics.solid_diffusion ∈ (:quadratic, :polynomial)
        residuals_c_s_avg_polynomial!(res, states, ∂states, p)
    else
        if     p.numerics.Fickian_method === :finite_difference # Use the FDM method for the solid phase diffusion
            residuals_c_s_avg_Fickian_FDM!(res, states, ∂states, p)
        elseif p.numerics.Fickian_method === :spectral # Use the spectral method for the discretization of the solid phase diffusion
            residuals_c_s_avg_Fickian_spectral!(res, states, ∂states, p)
        end
    end
    return nothing
end

function residuals_c_s_avg_polynomial!(res, states, ∂states, p)
    j = states[:j]
        
    ∂c_s_avg = ∂states[:c_s_avg]

    res_c_s_avg = res[:c_s_avg]
    
    # Cathode
    rhsCs_p = -3j.p/p.θ[:Rp_p]

    # Anode
    rhsCs_n = -3j.n/p.θ[:Rp_n]

    res_c_s_avg .= [rhsCs_p; rhsCs_n] .- ∂c_s_avg

    return nothing
end

function residuals_c_s_avg_Fickian_FDM!(res, states, ∂states, p::AbstractParam)
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

function residuals_c_s_avg_Fickian_spectral!(res, states, ∂states, p::AbstractParam)
    """
    Calculate the volume-averaged solid particle concentration residuals using a spectral method [mol/m³]
    """
    j = states[:j]
    c_s_avg = states[:c_s_avg]

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
    rhsCs_p = eltype(j.p)[]
    rhsCs_n = eltype(j.p)[]

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

function residuals_Q!(res, states, ∂states, p::AbstractParam)
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

function residuals_film!(res, states, ∂states, p::AbstractParam)
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
    T         = states[:T]
    I_density = states[:I][1]
    Q_rev     = states[:Q_rev]
    Q_rxn     = states[:Q_rxn]
    Q_ohm     = states[:Q_ohm]
    
    Q_rev_p = Q_rev.p
    Q_rev_n = Q_rev.n
    Q_rxn_p = Q_rxn.p
    Q_rxn_n = Q_rxn.n
    Q_ohm_p = Q_ohm.p
    Q_ohm_s = Q_ohm.s
    Q_ohm_n = Q_ohm.n

    ∂T = ∂states[:T]

    res_T = res[:T]

    Δx = Δx_values(p.N)

    T_BC_sx =  p.θ[:h_cell]*(p.θ[:T_amb]-T[1])/(Δx.a*p.θ[:l_a])
    T_BC_dx = -p.θ[:h_cell]*(T[end]-p.θ[:T_amb])/(Δx.z*p.θ[:l_z])

    block_tridiag(N) = Tridiagonal{eltype(I_density)}(ones(N-1),-[1;2ones(N-1)],ones(N-1))

    # Positive current collector
    A_a = p.θ[:λ_a].*block_tridiag(p.N.a)

    # Positive electrode
    A_p = p.θ[:λ_p].*block_tridiag(p.N.p)

    # Separator
    A_s = p.θ[:λ_s].*block_tridiag(p.N.s)

    # Negative electrode
    A_n = p.θ[:λ_n].*block_tridiag(p.N.n)

    # Negative current collector
    A_z = p.θ[:λ_z].*block_tridiag(p.N.z)

    A_z[end, end-1:end] = [p.θ[:λ_z] -p.θ[:λ_z]]

    # Divide the matrices by (Δx*l)^2

    A_a /= (Δx.a*p.θ[:l_a])^2
    A_p /= (Δx.p*p.θ[:l_p])^2
    A_s /= (Δx.s*p.θ[:l_s])^2
    A_n /= (Δx.n*p.θ[:l_n])^2
    A_z /= (Δx.z*p.θ[:l_z])^2

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
    β_a_p   = (Δx.a*p.θ[:l_a]/2)/(Δx.a*p.θ[:l_a]/2+Δx.p*p.θ[:l_p]/2)
    λ_a_p   = harmonic_mean(β_a_p,p.θ[:λ_a],p.θ[:λ_p])

    den_a_p = Δx.p*p.θ[:l_p]/2 +Δx.a*p.θ[:l_a]/2
    last_a  = p.θ[:λ_a] / (Δx.a*p.θ[:l_a])
    first_p = λ_a_p/den_a_p

    A_tot[p.N.a,p.N.a-1:p.N.a+1] .= [last_a; -(last_a+first_p); first_p]/(Δx.a*p.θ[:l_a])

    # Interface between aluminium current collector & positive electrode. We
    # are in the first volume of the positive electrode

    den_a_p  = Δx.p*p.θ[:l_p]/2 +Δx.a*p.θ[:l_a]/2
    second_p = p.θ[:λ_p] / (Δx.p*p.θ[:l_p])
    first_p  = λ_a_p/den_a_p

    A_tot[p.N.a+1,p.N.a:p.N.a+2] .= [first_p; -(second_p+first_p); second_p]/(Δx.p*p.θ[:l_p])

    # Interface between positive electrode & separator. We
    # are in the last volume of the positive electrode
    β_p_s = (Δx.p*p.θ[:l_p]/2)/(Δx.s*p.θ[:l_s]/2+Δx.p*p.θ[:l_p]/2)
    λ_p_s = harmonic_mean(β_p_s,p.θ[:λ_p],p.θ[:λ_s])

    den_p_s = Δx.p*p.θ[:l_p]/2 +Δx.s*p.θ[:l_s]/2
    last_p  = p.θ[:λ_p] / (Δx.p*p.θ[:l_p])
    first_s = λ_p_s/den_p_s

    A_tot[p.N.a+p.N.p,p.N.a+p.N.p-1:p.N.a+p.N.p+1] .= [last_p; -(last_p+first_s); first_s]/(Δx.p*p.θ[:l_p])

    # Interface between positive electrode & separator. We
    # are in the first volume of the separator
    den_p_s  = Δx.p*p.θ[:l_p]/2 +Δx.s*p.θ[:l_s]/2
    second_s = p.θ[:λ_s] / (Δx.s*p.θ[:l_s])
    first_s  = λ_p_s/den_p_s

    A_tot[p.N.a+p.N.p+1,p.N.a+p.N.p:p.N.a+p.N.p+2] .= [first_s; -(second_s+first_s); second_s]/(Δx.s*p.θ[:l_s])

    # Interface between separator negative electrode. We
    # are in the last volume of the separator
    β_s_n = (Δx.s*p.θ[:l_s]/2)/(Δx.s*p.θ[:l_s]/2+Δx.n*p.θ[:l_n]/2)
    λ_s_n = harmonic_mean(β_s_n,p.θ[:λ_s],p.θ[:λ_n])

    den_s_n = Δx.n*p.θ[:l_n]/2 +Δx.s*p.θ[:l_s]/2
    last_s  = p.θ[:λ_s] / (Δx.s*p.θ[:l_s])
    first_n = λ_s_n/den_s_n

    A_tot[p.N.a+p.N.p+p.N.s,p.N.a+p.N.p+p.N.s-1:p.N.a+p.N.p+p.N.s+1] .= [last_s; -(last_s+first_n); first_n]/(Δx.s*p.θ[:l_s])

    # Interface between separator negative electrode. We
    # are in the first volume of the negative electrode

    den_s_n  = Δx.n*p.θ[:l_n]/2 +Δx.s*p.θ[:l_s]/2
    second_n = p.θ[:λ_n] / (Δx.n*p.θ[:l_n])
    first_n  = λ_s_n/den_s_n

    A_tot[p.N.a+p.N.p+p.N.s+1,p.N.a+p.N.p+p.N.s:p.N.a+p.N.p+p.N.s+2] .= [first_n; -(first_n+second_n); second_n]/(Δx.n*p.θ[:l_n])


    # Interface between negative electrode & negative current collector. We
    # are in the last volume of the negative electrode
    β_n_z = (Δx.n*p.θ[:l_n]/2)/(Δx.z*p.θ[:l_z]/2+Δx.n*p.θ[:l_n]/2)
    λ_n_z = harmonic_mean(β_n_z,p.θ[:λ_n],p.θ[:λ_z])

    den_n_co = Δx.n*p.θ[:l_n]/2 +Δx.z*p.θ[:l_z]/2
    last_n   = p.θ[:λ_n] / (Δx.n*p.θ[:l_n])
    first_co = λ_n_z/den_n_co

    A_tot[p.N.a+p.N.p+p.N.s+p.N.n,p.N.a+p.N.p+p.N.s+p.N.n-1:p.N.a+p.N.p+p.N.s+p.N.n+1] .= [last_n; -(last_n+first_co); first_co]/(Δx.n*p.θ[:l_n])


    # Interface between negative electrode & negative current collector. We
    # are in the first volume of the negative current collector

    den_n_co  = Δx.n*p.θ[:l_n]/2 +Δx.z*p.θ[:l_z]/2
    second_co = p.θ[:λ_z] / (Δx.z*p.θ[:l_z])
    first_co  = λ_n_z/den_n_co

    A_tot[p.N.a+p.N.p+p.N.s+p.N.n+1,p.N.a+p.N.p+p.N.s+p.N.n:p.N.a+p.N.p+p.N.s+p.N.n+2] .= [first_co; -(second_co+first_co); second_co]/(Δx.z*p.θ[:l_z])

    Q_rev_tot = [
        zeros(p.N.a)
        Q_rev_p
        zeros(p.N.s)
        Q_rev_n
        zeros(p.N.z)
        ]

    Q_rxn_tot = [
        zeros(p.N.a)
        Q_rxn_p
        zeros(p.N.s)
        Q_rxn_n
        zeros(p.N.z)
        ]

    Q_ohm_tot = [
        (I_density^2)./repeat([p.θ[:σ_a]], p.N.a)
        Q_ohm_p
        Q_ohm_s
        Q_ohm_n
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

    rhsT   = A_tot*T
    rhsT .+= Q_rev_tot
    rhsT .+= Q_rxn_tot
    rhsT .+= Q_ohm_tot
    rhsT .+= BC
    rhsT ./= ρ_Cp

    res_T .= rhsT .- ∂T

    return nothing
end

function residuals_j!(res, states, p::AbstractParam)
    """
    Calculate the molar flux density of Li-ions residuals at the electrode-electrolyte interface [mol/(m²•s)]
    """
    c_s_star = states[:c_s_star]
    c_e = states[:c_e]
    T = states[:T]
    j = states[:j_orig]

    η = states[:η]

    res_j = res[:j]

    # Calculate the reaction rates
    k_p_eff, k_n_eff = coeff_reaction_rate(states, p)

    # only use for MHC reaction rate
    λ_MHC_p = haskey(p.θ, :λ_MHC_p) ? p.θ[:λ_MHC_p] : 0
    λ_MHC_n = haskey(p.θ, :λ_MHC_n) ? p.θ[:λ_MHC_n] : 0
    
    j_p_calc = p.numerics.rxn_p(c_s_star.p, c_e.p, T.p, η.p, k_p_eff, λ_MHC_p, p.θ[:c_max_p], p)
    j_n_calc = p.numerics.rxn_n(c_s_star.n, c_e.n, T.n, η.n, k_n_eff, λ_MHC_n, p.θ[:c_max_n], p)

    res_j .= [j_p_calc; j_n_calc] .- j
    
    return nothing
end

function residuals_j_s!(res, states, p::AbstractParam)
    """
    Calculate the molar flux density side reaction residuals due to SEI resistance [mol/(m²•s)]
    """
    j_s = states[:j_s]
    j   = states[:j_aging]
    Φ_s = states[:Φ_s]
    Φ_e = states[:Φ_e]
    film = states[:film]
    T = states[:T]
    I_density = states[:I][1]

    res_j_s = res[:j_s]

    F = const_Faradays
    R = const_Ideal_Gas
    
    I1C = calc_I1C(p)
    
    η_s = Φ_s.n .- Φ_e.n .- p.θ[:Uref_s] .- F.*j.n.*(p.θ[:R_SEI] .+ film./p.θ[:k_n_aging])
    
    # If aging is enabled; take into account the SEI resistance
    j_s_calc = -(p.θ[:i_0_jside].*(I_density/I1C)^p.θ[:w]./F).*exp.(-0.5F./(R.*T.n).*η_s)

    # side reaction residuals
    
    res_j_s .= [IfElse.ifelse(I_density > 0, j_s[i] .- j_s_calc[i], j_s[i]) for i in eachindex(j_s)]
    
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

    Δx = Δx_values(p.N)
    a_p, a_n = surface_area_to_volume_ratio(p)
    
    R = const_Ideal_Gas
    F = const_Faradays

    # Since the values of K_eff are evaluated at the c_enter of each CV; there is the need to interpolate these quantities
    # & find their values at the edges of the CVs
    K̂_eff_p, K̂_eff_s, K̂_eff_n = interpolate_electrolyte_conductivities(K_eff.p, K_eff.s, K_eff.n, p)

    A_tot = block_matrix_maker(p, K̂_eff_p, K̂_eff_s, K̂_eff_n)

    # dividing by the length and Δx
    A_tot[(1:p.N.p), (1:p.N.p)] ./= (Δx.p*p.θ[:l_p])
    A_tot[(1:p.N.s) .+ (p.N.p), (1:p.N.s) .+ (p.N.p)] ./= (Δx.s*p.θ[:l_s])
    A_tot[(1:p.N.n) .+ (p.N.p + p.N.s), (1:p.N.n) .+ (p.N.p + p.N.s)] ./= (Δx.n*p.θ[:l_n])

    # Fix values to enforce BC on the left side of the positive electrode.
    A_tot[1,1] = K̂_eff_p[1]./(Δx.p*p.θ[:l_p])
    A_tot[1,2] = -K̂_eff_p[1]./(Δx.p*p.θ[:l_p])

    # The value of Φ_e in the last volume of the negative electrode is known
    # & fixed.
    # right now, we have -K_eff [last interior face] + K_eff (last interior
    # face)
    
    A_tot[end, end-1:end] .= [0.0, 1.0]
    ## Interfaces Positive electrode [last volume of the positive]

    # Here we are in the last volume of the positive
    den = (Δx.p*p.θ[:l_p]/2+Δx.s*p.θ[:l_s]/2)
    last_p = K̂_eff_p[end-1]/(Δx.p*p.θ[:l_p])
    A_tot[p.N.p,p.N.p-1:p.N.p+1] = [-last_p, (last_p+K̂_eff_p[end]/den), -K̂_eff_p[end]/den]
    ## Interfaces Positive electrode [first volume of the separator]

    # Here we are in the first volume of the separator
    first_s = K̂_eff_s[1]/(Δx.s*p.θ[:l_s])
    A_tot[p.N.p+1,p.N.p:p.N.p+2] = [-K̂_eff_p[end]/den, (first_s+K̂_eff_p[end]/den), -first_s]

    ## Interfaces Positive electrode [last volume of the separator]
    # Here we are in the last volume of the separator
    den = (Δx.n*p.θ[:l_n]/2+Δx.s*p.θ[:l_s]/2)
    last_s = K̂_eff_s[end-1]/(Δx.s*p.θ[:l_s])
    A_tot[p.N.p+p.N.s,p.N.p+p.N.s-1:p.N.p+p.N.s+1] = [-last_s, (last_s+K̂_eff_s[end]/den), -K̂_eff_s[end]/den]

    ## Interfaces Positive electrode [first volume of the negative]
    # Here we are inside the first volume of the negative electrode
    first_n = K̂_eff_n[1]/(Δx.n*p.θ[:l_n])
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
    ν_p,ν_s,ν_n = p.numerics.thermodynamic_factor(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)
    ν = [ν_p;ν_s;ν_n]

    K = 2R.*(1-p.θ[:t₊]).*ν[1:end-1]/F

    prod_tot = [
        K̂_eff_p.*[T̄_p; T̄_ps].*[c_e_flux_p; c_e_flux_ps].*[1.0./c̄_e_p; 1.0./c̄_e_ps] # p
        K̂_eff_s.*[T̄_s; T̄_sn].*[c_e_flux_s; c_e_flux_sn].*[1.0./c̄_e_s; 1.0./c̄_e_sn] # s
        K̂_eff_n[1:end-1].*T̄_n.*c_e_flux_n./c̄_e_n # n
    ]
    prop_p_1 = prod_tot[1]

    prod_tot = prod_tot[2:end] .- prod_tot[1:end-1]
    prepend!(prod_tot, prop_p_1)

    f = -K.*prod_tot

    ind_p = (1:p.N.p)
    ind_n = (1:p.N.n) .+ (p.N.p+p.N.s)

    f[ind_p] .+= (Δx.p*p.θ[:l_p]*F*a_p)*j.p
    f[ind_n[1:end-1]] .+= (Δx.n*p.θ[:l_n]*F*a_n)*j.n[1:end-1]

    # Set the last element of Φ_e to 0 [enforcing BC]
    append!(f, 0.0)

    # Return the residual value for the electrolyte potential
    res_Φ_e .= A_tot*Φ_e .- f

    return nothing
end

function residuals_Φ_s!(res, states, p::AbstractParam)
    """
    residuals_Φ_s! evaluates the residuals of the solid potential equation [V]
    """

    j = states[:j_aging]
    Φ_s = states[:Φ_s]
    I_density = states[:I][1]

    res_Φ_s = res[:Φ_s]

    F = const_Faradays
    Δx = Δx_values(p.N)
    
    ## Positive electrode

    # RHS for the solid potential in the positive electrode. The BC on the left is enforced [Neumann BC]
    σ_eff_p, σ_eff_n = conductivity_effective(p)
    a_p, a_n = surface_area_to_volume_ratio(p)

    @views @inbounds f_p = [
        ((p.θ[:l_p].*Δx.p.*a_p.*F.*j.p[1])-I_density).*Δx.p.*p.θ[:l_p]./σ_eff_p
        (p.θ[:l_p].^2 .* Δx.p.^2 .* a_p.*F.*j.p[2:end])./σ_eff_p
    ]
    
    # RHS for the solid potential in the positive electrode.

    ## Negative electrode

    # RHS for the solid potential in the negative electrode.
    @views @inbounds f_n = (p.θ[:l_n].^2 .* Δx.n.^2 .* a_n.*F.*j.n[1:end-1])./σ_eff_n

    # RHS for the solid potential in the negative electrode.
    append!(f_n, ((p.θ[:l_n].*Δx.n.*a_n.*F.*j.n[end])+I_density).*Δx.n.*p.θ[:l_n]./σ_eff_n)

    block_tridiag(N) = Tridiagonal{eltype(j.p)}(ones(N-1),-[1;2ones(N-2);1],ones(N-1))
    ## Residual array
    # Return the residual array
    A_p = block_tridiag(p.N.p)
    A_n = block_tridiag(p.N.n)

    A_n *= Φ_s.n
    A_n .-= f_n
    
    A_p *= Φ_s.p
    A_p .-= f_p

    res_Φ_s .= [A_p; A_n]

    return nothing
end

function residuals_scalar!(res, states, p::AbstractParam)
    """
    *** THE RESIDUALS ARE HANDLED IN `scalar_residual.jl` ***
    
    This function sets an arbitrary value for the current residuals.
    """
    
    res_I = res[:I]

    res_I .= 0.0
    
    return nothing
end
