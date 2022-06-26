"""
Much of the code in this file is adapted from LIONSIMBA, a li-ion simulation toolbox in MATLAB.
See https://github.com/lionsimbatoolbox/LIONSIMBA for more information.
"""

function residuals_PET!(residuals, t, x, ẋ, p::AbstractModel)
    """
    First put the vector of x, ẋ, and residuals into dictionaries
    """
    states  = retrieve_states(x, p)
    ∂states = retrieve_states(ẋ, p)
    res     = retrieve_states(residuals, p)

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
    
    # If the polynomial fit for the surface concentration is used, Q
    if p.numerics.solid_diffusion == :polynomial
        residuals_Q!(res, states, ∂states, p)
    end

    # Lithium plating film thickness, film
    if p.numerics.aging == :SEI
        residuals_film!(res, states, ∂states, p)
        residuals_SOH!(res, states, ∂states, p)
    end

    # Check if the thermal dynamics are enabled.
    if p.numerics.temperature == true
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

    # Residuals for applied current density, I [not currently used]
    residuals_scalar!(res, states, p)

    """
    Compile all residuals together
    """
    build_residuals!(residuals, res, p)

    return nothing
end
function residuals_PET!(p::AbstractModel)
    θ_sym, Y_sym, YP_sym, t_sym, SOC_sym, X_applied, γ_sym, p_sym, θ_keys = get_symbolic_vars(p)
    res = similar(Y_sym)
    residuals_PET!(res, t_sym, Y_sym, YP_sym, p_sym)
end

function residuals_c_e!(res, states, ∂states, p::AbstractModel)
    """
    Calculate the electrolyte concentration residuals [mol/m³]
    """

    c_e = states[:c_e]
    T   = states[:T]
    j   = states[:j_total]

    ∂c_e = ∂states[:c_e]

    res_c_e = res[:c_e]

    Δx = Δx_values(p.N)

    # Diffusion coefficients and surface area to volume ratio
    D_eff_p, D_eff_s, D_eff_n = coeff_electrolyte_diffusion_effective(states, p)
    a_p, a_n = surface_area_to_volume_ratio(p)

    # Interpolation of the diffusion coefficients, same for electrolyte conductivities
    D_eff_p, D_eff_s, D_eff_n = interpolate_electrolyte_conductivities(D_eff_p, D_eff_s, D_eff_n, p)

    A_tot = -block_matrix_maker(p, D_eff_p, D_eff_s, D_eff_n)

    # dividing by the length and Δx
    ind = 1:p.N.p
    @views @inbounds A_tot[ind,ind] ./= (Δx.p*p.θ[:l_p])^2
    ind = (1:p.N.s) .+ ind[end]
    @views @inbounds A_tot[ind,ind] ./= (Δx.s*p.θ[:l_s])^2
    ind = (1:p.N.n) .+ ind[end]
    @views @inbounds A_tot[ind,ind] ./= (Δx.n*p.θ[:l_n])^2

    # Reset values on the lines for the interfaces conditions
    @views @inbounds A_tot[p.N.p,:]   .= 0.0
    @views @inbounds A_tot[p.N.p+1,:] .= 0.0

    # Reset values on the lines for the interfaces conditions
    @views @inbounds A_tot[p.N.p+p.N.s,:]   .= 0.0
    @views @inbounds A_tot[p.N.p+p.N.s+1,:] .= 0.0

    ## Interface between separator and cathode (last volume in the cathode)

    # Compute the common denominator at the interface
    @views @inbounds den_s = (Δx.p*p.θ[:l_p]/2 + Δx.s*p.θ[:l_s]/2)
    # Last diffusion coefficient of the cathode
    @views @inbounds last_p = D_eff_p[end-1]/(Δx.p*p.θ[:l_p])
    # Diffusion coefficient on the interface
    @views @inbounds first_s = D_eff_p[end]/den_s
    # Fix the values at the boundaries
    @views @inbounds A_tot[p.N.p,p.N.p-1:p.N.p+1] .= [last_p; -(last_p + first_s); first_s]/(Δx.p*p.θ[:l_p]*p.θ[:ϵ_p])

    ## Interface between separator and cathode (first volume in the separator)

    # First diffusion coefficient in the separator
    @views @inbounds second_s = D_eff_s[1]/(Δx.s*p.θ[:l_s])
    # Diffusion coefficient on the interface
    @views @inbounds first_s = D_eff_p[end]/den_s

    @views @inbounds A_tot[p.N.p+1,p.N.p:p.N.p+2] .= [first_s; -(first_s+second_s); second_s]/(Δx.s*p.θ[:l_s]*p.θ[:ϵ_s])

    ## Interface between separator and anode (last volume in the separator)

    # Compute the common denominator at the interface
    @views @inbounds den_s = (Δx.s*p.θ[:l_s]/2 + Δx.n*p.θ[:l_n]/2)
    # Last diffusion coefficient in the separator
    @views @inbounds last_s = D_eff_s[end-1]/(Δx.s*p.θ[:l_s])
    # Diffusion coefficient on the interface
    @views @inbounds first_n = D_eff_s[end]/den_s

    @views @inbounds A_tot[p.N.p+p.N.s,p.N.p+p.N.s-1:p.N.p+p.N.s+1] .= [last_s; -(last_s+first_n); first_n]/(Δx.s*p.θ[:l_s]*p.θ[:ϵ_s])

    ## Interface between separator and anode (first volume in the anode)

    # Compute the common denominator at the interface
    den_n = (Δx.s*p.θ[:l_s]/2 + Δx.n*p.θ[:l_n]/2)
    # First diffusion coefficient in the anode
    @views @inbounds second_n = D_eff_n[1]/(Δx.n*p.θ[:l_n])
    # Diffusion coefficient on the interface
    @views @inbounds first_n = D_eff_s[end]/den_n

    
    A_tot[p.N.p+p.N.s+1,p.N.p+p.N.s:p.N.p+p.N.s+2] .= [first_n; -(first_n+second_n); second_n]/(Δx.n*p.θ[:l_n]*p.θ[:ϵ_n])
    
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

    A_tot = Matrix(A_tot).*Matrix(A_ϵ)

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

function residuals_c_s_avg!(res, states, ∂states, p::T) where {T<:Union{AbstractModelSolidDiff{:polynomial},AbstractModelSolidDiff{:quadratic}}}
    """
    Calculate the solid particle concentrations residuals with quadratic or polynomial approximations [mol/m³]
    """
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
function residuals_c_s_avg!(res, states, ∂states, p::T) where {jac,temp,T<:AbstractModel{jac,temp,:Fickian,:finite_difference}}
    """
    Calculate the volume-averaged solid particle concentration residuals using a 9th order accurate finite difference method (FDM) [mol/m³]
    """
    j = states[:j]
    c_s_avg = states[:c_s_avg]

    ∂c_s_avg = ∂states[:c_s_avg]

    res_c_s_avg = res[:c_s_avg]

    # First, retreive the diffusion coefficients
    D_sp_eff, D_sn_eff = coeff_solid_diffusion_effective(states, p)

    function rhs_func(c_s_avg,j,Rp,D_s_eff,N,N_r)
        # Matrices needed for first and second order derivatives
        deriv = derivative_matrices_first_and_second_order(N_r)

        rhsCs = zeros(eltype(j), N*N_r)
        @inbounds for i in 1:N
            ind = (1:N_r) .+ N_r*(i-1)
            c_s = @inbounds @views c_s_avg[ind]
            
            # First order derivatives matrix multiplication
            ∂ᵣc_s = deriv[1](c_s)
    
            # Boundary condition at r = 1
            ∂ᵣc_s[N_r] = -j[i]/D_s_eff[i]*Rp
    
            # Boundary condition at r = 0
            ∂ᵣc_s[1] = 0
    
            # Second order derivatives matrix multiplication
            ∂ᵣᵣc_s = deriv[2](c_s)
    
            # Neumann BC at r = 1
            ∂ᵣᵣc_s[end] += 50deriv[2].Δx*∂ᵣc_s[N_r]*(deriv[2].coeff)
    
            # Make the RHS vector for this particle
            @inbounds rhsCs[ind] .= (D_s_eff[i]./Rp^2) .* [
                3∂ᵣᵣc_s[1]
                ∂ᵣᵣc_s[2:end]+2.0./range(1/(N_r-1), 1, length=N_r-1).*∂ᵣc_s[2:end]
                ]

        end
        return rhsCs
    end

    rhsCs_p = rhs_func(c_s_avg.p,j.p,p.θ[:Rp_p],D_sp_eff,p.N.p,p.N.r_p)
    rhsCs_n = rhs_func(c_s_avg.n,j.n,p.θ[:Rp_n],D_sn_eff,p.N.n,p.N.r_n)
    
    res_c_s_avg .= [rhsCs_p; rhsCs_n] .- ∂c_s_avg

    return nothing
end
function residuals_c_s_avg!(res, states, ∂states, p::T) where {jac,temp,T<:AbstractModel{jac,temp,:Fickian,:spectral}}
    """
    BETA: Calculate the volume-averaged solid particle concentration residuals using a spectral method [mol/m³]
    """
    j = states[:j]
    c_s_avg = states[:c_s_avg]

    ∂c_s_avg = ∂states[:c_s_avg]

    res_c_s_avg = res[:c_s_avg]

    function cheb(N::Int64)
        x = cos.(range(0,π,length=N+1))
        c = [2.0; ones(N-1); 2.0].*(-1).^(0:N)

        X = repeat(x,1,N+1)
        dX = X .- X'
        D = (c*(1.0./c)')./(dX .+ LinearAlgebra.I(N+1)) # off-diagonal entries
        D[diagind(D)] .-= sum(D', dims=1)[:]

        return D, x
    end
    
    # First, retreive the diffusion coefficients
    D_sp_eff, D_sn_eff = coeff_solid_diffusion_effective(states, p)

    function rhs_func(c_s_avg,j,Rp,D_s_eff,N,N_r)
        diffusion_matrix, radial_position = cheb(N_r-1)

        rhsCs = zeros(eltype(j),N*N_r)
        @inbounds for i in 1:N
            ind = ind = (1:N_r) .+ N_r*(i-1)
            c_s = c_s_avg[ind]
            
            ∂ᵣc_s = diffusion_matrix*reverse(c_s)
            ∂ᵣc_s[1] = -j[i]*Rp*0.5/D_s_eff[i] # modified BC value due to cheb scheme
            ∂ᵣc_s[end] = 0 # no flux BC
            
            rhs_numerator = reverse(diffusion_matrix*(4*D_s_eff[i]*((radial_position .+ 1).^2).*∂ᵣc_s/(Rp^2)))
            
            rhs_limit_vector = (4*D_s_eff[i]/Rp^2)*3*(diffusion_matrix*∂ᵣc_s) # limit at r_tilde tends to -1 (at center)
            
            @inbounds rhsCs[ind] .= [
                rhs_limit_vector[end] # L'hopital's rule at the center of the particle
                rhs_numerator[2:end]./((reverse(radial_position[1:end-1]) .+ 1).^2)
                ]
        end
        return rhsCs
    end

    rhsCs_p = rhs_func(c_s_avg.p,j.p,p.θ[:Rp_p],D_sp_eff,p.N.p,p.N.r_p)
    rhsCs_n = rhs_func(c_s_avg.n,j.n,p.θ[:Rp_n],D_sn_eff,p.N.n,p.N.r_n)
    
    res_c_s_avg .= [rhsCs_p; rhsCs_n] .- ∂c_s_avg
    
    return nothing
end

function residuals_Q!(res, states, ∂states, p::AbstractModelSolidDiff{:polynomial})
    """
    residuals_Q! is used to implement the three parameters reduced sol for solid phase diffusion [mol/m⁴]
    This sol has been taken from the paper "Efficient Macro-Micro Scale Coupled
    Modeling of Batteries" by Subramanian et al.
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

function residuals_film!(res, states, ∂states, p::AbstractModel)
    """
    residuals_film! describes the dynamics of the solid-electrolyte layer at the anode side [m]
    """

    j_s = states[:j_s]

    ∂film = ∂states[:film]
    
    res_film = res[:film]
    
    rhs_film = -j_s.*p.θ[:M_n]./p.θ[:ρ_n]

    res_film .= rhs_film .- ∂film

    return nothing
end

function residuals_SOH!(res, states, ∂states, p::AbstractModel)
    """
    residuals_SOH! integrates the SOH when aging is enabled and there are losses
    """

    j_s = states[:j_s]
    
    ∂SOH = ∂states[:SOH][1]

    res_SOH = res[:SOH]

    j_s_int = -trapz(extrapolate_section(j_s, p, :n)...)
    j_s_int *= const_Faradays*surface_area_to_volume_ratio(p)[2]/(3600*calc_I1C(p.θ))

    rhs_SOH = -j_s_int

    res_SOH .= rhs_SOH - ∂SOH

    return nothing
end

function residuals_T!(res, states, ∂states, p)
    """
    Calculate the 1D temperature residuals [K]
    """
    T         = states[:T]
    I_density = states[:I][1]
    Q_rev     = states[:Q_rev]
    Q_rxn     = states[:Q_rxn]
    Q_ohm     = states[:Q_ohm]

    ∂T = ∂states[:T]

    res_T = res[:T]

    Δx = Δx_values(p.N)

    T_BC_sx =  p.θ[:h_cell]*(p.θ[:T_amb]-T[1])/(Δx.a*p.θ[:l_a])
    T_BC_dx = -p.θ[:h_cell]*(T[end]-p.θ[:T_amb])/(Δx.z*p.θ[:l_z])

    block_tridiag(N) = spdiagm(
        -1 => ones(eltype(I_density),N-1),
        0 => -[1;2ones(eltype(I_density),N-2);1],
        +1 => ones(eltype(I_density),N-1),
        )

    # Positive current collector
    A_a = p.θ[:λ_a].*block_tridiag(p.N.a)

    # Cathode
    A_p = p.θ[:λ_p].*block_tridiag(p.N.p)

    # Separator
    A_s = p.θ[:λ_s].*block_tridiag(p.N.s)

    # Anode
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

    # Interface between aluminium current collector & cathode. We
    # are in the last volume of the current collector
    β_a_p   = (Δx.a*p.θ[:l_a]/2)/(Δx.a*p.θ[:l_a]/2+Δx.p*p.θ[:l_p]/2)
    λ_a_p   = harmonic_mean(β_a_p,p.θ[:λ_a],p.θ[:λ_p])

    den_a_p = Δx.p*p.θ[:l_p]/2 +Δx.a*p.θ[:l_a]/2
    last_a  = p.θ[:λ_a] / (Δx.a*p.θ[:l_a])
    first_p = λ_a_p/den_a_p

    A_tot[p.N.a,p.N.a-1:p.N.a+1] .= [last_a; -(last_a+first_p); first_p]/(Δx.a*p.θ[:l_a])

    # Interface between aluminium current collector & cathode. We
    # are in the first volume of the cathode

    den_a_p  = Δx.p*p.θ[:l_p]/2 +Δx.a*p.θ[:l_a]/2
    second_p = p.θ[:λ_p] / (Δx.p*p.θ[:l_p])
    first_p  = λ_a_p/den_a_p

    A_tot[p.N.a+1,p.N.a:p.N.a+2] .= [first_p; -(second_p+first_p); second_p]/(Δx.p*p.θ[:l_p])

    # Interface between cathode & separator. We
    # are in the last volume of the cathode
    β_p_s = (Δx.p*p.θ[:l_p]/2)/(Δx.s*p.θ[:l_s]/2+Δx.p*p.θ[:l_p]/2)
    λ_p_s = harmonic_mean(β_p_s,p.θ[:λ_p],p.θ[:λ_s])

    den_p_s = Δx.p*p.θ[:l_p]/2 +Δx.s*p.θ[:l_s]/2
    last_p  = p.θ[:λ_p] / (Δx.p*p.θ[:l_p])
    first_s = λ_p_s/den_p_s

    A_tot[p.N.a+p.N.p,p.N.a+p.N.p-1:p.N.a+p.N.p+1] .= [last_p; -(last_p+first_s); first_s]/(Δx.p*p.θ[:l_p])

    # Interface between cathode & separator. We
    # are in the first volume of the separator
    den_p_s  = Δx.p*p.θ[:l_p]/2 +Δx.s*p.θ[:l_s]/2
    second_s = p.θ[:λ_s] / (Δx.s*p.θ[:l_s])
    first_s  = λ_p_s/den_p_s

    A_tot[p.N.a+p.N.p+1,p.N.a+p.N.p:p.N.a+p.N.p+2] .= [first_s; -(second_s+first_s); second_s]/(Δx.s*p.θ[:l_s])

    # Interface between separator anode. We
    # are in the last volume of the separator
    β_s_n = (Δx.s*p.θ[:l_s]/2)/(Δx.s*p.θ[:l_s]/2+Δx.n*p.θ[:l_n]/2)
    λ_s_n = harmonic_mean(β_s_n,p.θ[:λ_s],p.θ[:λ_n])

    den_s_n = Δx.n*p.θ[:l_n]/2 +Δx.s*p.θ[:l_s]/2
    last_s  = p.θ[:λ_s] / (Δx.s*p.θ[:l_s])
    first_n = λ_s_n/den_s_n

    A_tot[p.N.a+p.N.p+p.N.s,p.N.a+p.N.p+p.N.s-1:p.N.a+p.N.p+p.N.s+1] .= [last_s; -(last_s+first_n); first_n]/(Δx.s*p.θ[:l_s])

    # Interface between separator anode. We
    # are in the first volume of the anode

    den_s_n  = Δx.n*p.θ[:l_n]/2 +Δx.s*p.θ[:l_s]/2
    second_n = p.θ[:λ_n] / (Δx.n*p.θ[:l_n])
    first_n  = λ_s_n/den_s_n

    A_tot[p.N.a+p.N.p+p.N.s+1,p.N.a+p.N.p+p.N.s:p.N.a+p.N.p+p.N.s+2] .= [first_n; -(first_n+second_n); second_n]/(Δx.n*p.θ[:l_n])


    # Interface between anode & negative current collector. We
    # are in the last volume of the anode
    β_n_z = (Δx.n*p.θ[:l_n]/2)/(Δx.z*p.θ[:l_z]/2+Δx.n*p.θ[:l_n]/2)
    λ_n_z = harmonic_mean(β_n_z,p.θ[:λ_n],p.θ[:λ_z])

    den_n_co = Δx.n*p.θ[:l_n]/2 +Δx.z*p.θ[:l_z]/2
    last_n   = p.θ[:λ_n] / (Δx.n*p.θ[:l_n])
    first_co = λ_n_z/den_n_co

    A_tot[p.N.a+p.N.p+p.N.s+p.N.n,p.N.a+p.N.p+p.N.s+p.N.n-1:p.N.a+p.N.p+p.N.s+p.N.n+1] .= [last_n; -(last_n+first_co); first_co]/(Δx.n*p.θ[:l_n])
    
    # Interface between anode & negative current collector. We
    # are in the first volume of the negative current collector

    den_n_co  = Δx.n*p.θ[:l_n]/2 +Δx.z*p.θ[:l_z]/2
    second_co = p.θ[:λ_z] / (Δx.z*p.θ[:l_z])
    first_co  = λ_n_z/den_n_co

    A_tot[p.N.a+p.N.p+p.N.s+p.N.n+1,p.N.a+p.N.p+p.N.s+p.N.n:p.N.a+p.N.p+p.N.s+p.N.n+2] .= [first_co; -(second_co+first_co); second_co]/(Δx.z*p.θ[:l_z])

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

    rhsT   = A_tot*T
    rhsT .+= Q_rev_tot
    rhsT .+= Q_rxn_tot
    rhsT .+= Q_ohm_tot
    rhsT .+= BC
    rhsT ./= ρ_Cp

    res_T .= rhsT .- ∂T

    return nothing
end

function residuals_j!(res, states, p::AbstractModel)
    """
    Calculate the molar flux density of Li-ions residuals at the electrode-electrolyte interface [mol/(m²•s)]
    """
    c_s_star = states[:c_s_star]
    c_e = states[:c_e]
    T = states[:T]
    j = states[:j]

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

function residuals_j_s!(res, states, p::AbstractModel)
    """
    Calculate the molar flux density side reaction residuals due to SEI resistance [mol/(m²•s)]
    """
    j_s = states[:j_s]
    Φ_s = states[:Φ_s]
    Φ_e = states[:Φ_e]
    film = states[:film]
    T = states[:T]
    I_density = states[:I][1]

    res_j_s = res[:j_s]

    F = const_Faradays
    R = const_Ideal_Gas
    
    I1C = calc_I1C(p)
    
    R_film = (p.θ[:R_SEI] .+ film./p.θ[:k_n_aging])

    η_s = Φ_s.n .- Φ_e.n .- p.θ[:Uref_s] .- F.*j_s.*R_film
    α = 0.5

    j_s_calc = -abs.((p.θ[:i_0_jside].*(I_density/I1C)^p.θ[:w]./F).*(-exp.(-α.*F./(R.*T.n).*η_s)))

    # Only activate the side reaction during charge
    j_s_calc .= IfElse.ifelse(I_density > 0, j_s_calc, 0)

    # side reaction residuals
    res_j_s .= j_s .- j_s_calc
    
    return nothing
end

function residuals_Φ_e!(res, states, p::AbstractModel)
    """
    residuals_Φ_e! evaluates residuals for the electrolyte potential equation discretized using method of lines, [V]
    """

    j   = states[:j_total]
    Φ_e = states[:Φ_e]
    c_e = states[:c_e]
    T   = states[:T]

    K_eff = states[:K_eff]

    res_Φ_e = res[:Φ_e]

    Δx = Δx_values(p.N)
    a_p, a_n = surface_area_to_volume_ratio(p)
    
    R = const_Ideal_Gas
    F = const_Faradays

    # Interpolate the K_eff values to the edge of the control volume
    K̂_eff_p, K̂_eff_s, K̂_eff_n = interpolate_electrolyte_conductivities(K_eff.p, K_eff.s, K_eff.n, p)

    A_tot = block_matrix_maker(p, K̂_eff_p, K̂_eff_s, K̂_eff_n)

    ind = 1:p.N.p
    A_tot[ind,ind] ./= (Δx.p*p.θ[:l_p])
    ind = (1:p.N.s) .+ ind[end]
    A_tot[ind,ind] ./= (Δx.s*p.θ[:l_s])
    ind = (1:p.N.n) .+ ind[end]
    A_tot[ind,ind] ./= (Δx.n*p.θ[:l_n])

    # Φ_e(x = L) = 0
    A_tot[end, end-1:end] .= [0.0, 1.0]

    ## Interfaces Cathode [last volume of the positive]

    # Here we are in the last volume of the positive
    den = (Δx.p*p.θ[:l_p]/2+Δx.s*p.θ[:l_s]/2)
    last_p = K̂_eff_p[end-1]/(Δx.p*p.θ[:l_p])
    A_tot[p.N.p,p.N.p-1:p.N.p+1] .= [-last_p, (last_p+K̂_eff_p[end]/den), -K̂_eff_p[end]/den]

    ## Interfaces Cathode [first volume of the separator]

    # Here we are in the first volume of the separator
    first_s = K̂_eff_s[1]/(Δx.s*p.θ[:l_s])
    A_tot[p.N.p+1,p.N.p:p.N.p+2] .= [-K̂_eff_p[end]/den, (first_s+K̂_eff_p[end]/den), -first_s]

    ## Interfaces Cathode [last volume of the separator]
    # Here we are in the last volume of the separator
    den = (Δx.n*p.θ[:l_n]/2+Δx.s*p.θ[:l_s]/2)
    last_s = K̂_eff_s[end-1]/(Δx.s*p.θ[:l_s])
    A_tot[p.N.p+p.N.s,p.N.p+p.N.s-1:p.N.p+p.N.s+1] .= [-last_s, (last_s+K̂_eff_s[end]/den), -K̂_eff_s[end]/den]

    ## Interfaces Cathode [first volume of the negative]
    # Here we are inside the first volume of the anode
    first_n = K̂_eff_n[1]/(Δx.n*p.θ[:l_n])
    A_tot[p.N.p+p.N.s+1,p.N.p+p.N.s:p.N.p+p.N.s+2] .= [-K̂_eff_s[end]/den, (first_n+K̂_eff_s[end]/den), -first_n]


    ## Electrolyte concentration interpolation
    # Evaluate the interpolation of the electrolyte concentration values at the
    # edges of the control volumes.
    c̄_e_p, c̄_e_s, c̄_e_n = PETLION.interpolate_electrolyte_concentration(c_e, p)
    ## Temperature interpolation
    # Evaluate the temperature value at the edges of the control volumes
    T̄_p, T̄_s, T̄_n = PETLION.interpolate_temperature(T, p)
    ## Electrolyte fluxes
    # Evaluate the interpolation of the electrolyte concentration fluxes at the
    # edges of the control volumes.
    ∂ₓc_e_p, ∂ₓc_e_s, ∂ₓc_e_n = PETLION.interpolate_electrolyte_concetration_fluxes(c_e, p)
    
    ## RHS arrays
    ν_p,ν_s,ν_n = p.numerics.thermodynamic_factor(c_e.p, c_e.s, c_e.n, T.p, T.s, T.n, p)
    ν = [ν_p;ν_s;ν_n]

    K = 2R.*(1-p.θ[:t₊]).*ν[1:end-1]/F

    prod_tot = [
        K̂_eff_p.*T̄_p.*∂ₓc_e_p./c̄_e_p # p
        K̂_eff_s.*T̄_s.*∂ₓc_e_s./c̄_e_s # s
        K̂_eff_n[1:end-1].*T̄_n.*∂ₓc_e_n./c̄_e_n # n
    ]
    
    prod_tot[2:end] .-= prod_tot[1:end-1]

    f = -K.*prod_tot

    ind_p = (1:p.N.p)
    ind_n = (1:p.N.n) .+ (p.N.p+p.N.s)

    f[ind_p] .+= (Δx.p*p.θ[:l_p]*F*a_p)*j.p
    f[ind_n[1:end-1]] .+= (Δx.n*p.θ[:l_n]*F*a_n)*j.n[1:end-1]

    # The boundary condition enforces that Φ_e(x=L) = 0
    append!(f, 0.0)

    # Return the residual value for the electrolyte potential
    res_Φ_e .= A_tot*Φ_e .- f

    return nothing
end

function residuals_Φ_s!(res, states, p::AbstractModel)
    """
    residuals_Φ_s! evaluates the residuals of the solid potential equation [V]
    """

    j = states[:j_total]
    Φ_s = states[:Φ_s]
    I_density = states[:I][1]

    res_Φ_s = res[:Φ_s]

    F = const_Faradays
    Δx = Δx_values(p.N)
    
    # Effective conductivities and surface area to volume ratio
    σ_eff_p, σ_eff_n = conductivity_effective(p)
    a_p, a_n = surface_area_to_volume_ratio(p)

    ## Cathode
    
    # RHS for the solid potential in the cathode and anode
    f_p = @. p.θ[:l_p]^2*Δx.p^2*a_p*F*j.p/σ_eff_p
    f_n = @. p.θ[:l_n]^2*Δx.n^2*a_n*F*j.n/σ_eff_n

    # Additional term at the electrode-current collector interface
    f_p[1]   += -I_density*(Δx.p*p.θ[:l_p]/σ_eff_p)
    f_n[end] += +I_density*(Δx.n*p.θ[:l_n]/σ_eff_n)

    block_tridiag(N) = spdiagm(
        -1 => ones(eltype(j.p),N-1),
        0 => -[1;2ones(eltype(j.p),N-2);1],
        +1 => ones(eltype(j.p),N-1),
        )
    
    A_p = block_tridiag(p.N.p)
    A_n = block_tridiag(p.N.n)

    A_n *= Φ_s.n
    A_n .-= f_n
    
    A_p *= Φ_s.p
    A_p .-= f_p

    res_Φ_s .= [A_p; A_n]

    return nothing
end

function residuals_scalar!(res, states, p::AbstractModel)
    """
    *** THE RESIDUALS ARE HANDLED IN `scalar_residual.jl` ***
    
    This function sets an arbitrary value for the current residuals [C-rate]
    """
    
    res_I = res[:I]

    res_I .= 0.0
    
    return nothing
end
