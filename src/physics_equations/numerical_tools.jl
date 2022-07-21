struct DiffusionMatrix
    M
    coeff
    Δx
end
(deriv::DiffusionMatrix)(x::AbstractVector) = deriv.coeff*(deriv.M*x)

function derivative_matrix_first_order(n, xl, xu)
    Δx = (xu - xl)/(n - 1)
    r8fΔx = 1/(40320Δx)

    first_block = [
        -109584.0 +322560 -564480 +752640 -705600 +451584 -188160 +46080 -5040
        -5040.0 -64224 +141120 -141120 +117600 -70560 +28224 -6720 +720
        +720.0 -11520 -38304 +80640 -50400 +26880 -10080 +2304 -240
        -240.0 +2880 -20160 -18144 +50400 -20160 +6720 -1440 +144
        ]
    first_block = [first_block zeros(Float64, 4, n-9)]
    
    i_th_row = [+144.0 -1536 +8064 -32256 +0 +32256 -8064 +1536 -144]
    
    last_block = [
        -144.0 +1440 -6720 +20160 -50400 +18144 +20160 -2880 +240
        +240.0 -2304 +10080 -26880 +50400 -80640 +38304 +11520 -720
        -720.0 +6720 -28224 +70560 -117600 +141120 -141120 +64224 +5040
        +5040.0 -46080 +188160 -451584 +705600 -752640 +564480 -322560 +109584
        ]
    last_block = [zeros(Float64, 4, n-9) last_block]

    mid_block = zeros(Float64, n-8, n)
    @inbounds for (row_index, i) in enumerate(5:n-4)
        @inbounds mid_block[row_index,row_index:row_index+8] = i_th_row
    end

    deriv_matrix = [
        first_block
        mid_block
        last_block
        ]

    return DiffusionMatrix(deriv_matrix, r8fΔx, Δx)
end

function derivative_matrix_second_order(n, xl, xu)
    Δx = (xu - xl)/(n - 1)

    r12Δxs = 1/(12Δx^2)

    mid_block = zeros(Float64, n - 4, n)

    first_block = [
        -415/6 +96 -36 +32/3 -3/2 0
        +10.0 -15 -4 +14 -6 +1
        ]
    first_block = [first_block zeros(Float64, 2, n-6)]

    i_th_row = [-1.0 +16 -30 +16 -1]
    
    last_block = [
        +1.0 -6 +14 -4 -15 +10
        0.0 -3/2 +32/3 -36 +96 -415/6
        ]
    last_block = [zeros(Float64, 2, n-6) last_block]

    @inbounds for (row_index, i) in enumerate(3:n-2)
        @inbounds mid_block[row_index, row_index:row_index+4] = i_th_row
    end

    deriv_matrix = [
        first_block
        mid_block
        last_block
        ]

    return DiffusionMatrix(deriv_matrix, r12Δxs, Δx)
end

function derivative_matrices_first_and_second_order(N, xl=0, xu=1)
    """
    Combines the first and second order derivative matrices into a dictionary
    """
    out = Dict{Int64,DiffusionMatrix}()
    out[1] = derivative_matrix_first_order(N, xl, xu)
    out[2] = derivative_matrix_second_order(N, xl, xu)

    return out
end

function block_matrix_maker(p, X, Y, Z)
    """
    Finite volume stencil for the electrolyte grid (cathode, separator, anode)
    """
    single_block(x) = Matrix(Tridiagonal{eltype(x)}(-x[1:end-1],x .+ [0;x[1:end-1]],-x[1:end-1]))

    ind = indices_section((:p,:s,:n), p)

    A_tot = zeros(eltype(X), ind.n[end], ind.n[end])

    A_tot[ind.p,ind.p] .= single_block(X)
    A_tot[ind.s,ind.s] .= single_block(Y)
    A_tot[ind.n,ind.n] .= single_block(Z)
    
    return A_tot
end

function interpolate_electrolyte_grid(Keff_p, Keff_s, Keff_n, p::AbstractModel)
    """
    interpolate_electrolyte_grid interpolates electrolyte conductivities at the edges of control volumes using harmonic mean.
    """

    Δx = Δx_values(p.N)

    Keff_i_medio(β_i, Keff_i) = Keff_i[1:end-1].*Keff_i[2:end]./(β_i*Keff_i[2:end]+(1-β_i)*Keff_i[1:end-1])
    β_i_j(Δx_i, l_i, Δx_j, l_j) = Δx_i*l_i/2 /(Δx_j*l_j/2+Δx_i*l_i/2)
    Keff_i_j_interface(β_i_j, Keff_i, Keff_j) = Keff_i[end]*Keff_j[1] / (β_i_j*Keff_j[1] + (1-β_i_j)*Keff_i[end])

    ## Positive electrode mean conductivity
    β_p = 0.5
    Keff_p_medio = Keff_i_medio(β_p, Keff_p)

    # The last element of Keff_p_medio will be the harmonic mean of the
    # elements at the interface positive-separator

    β_p_s = β_i_j(Δx.p, p.θ[:l_p], Δx.s, p.θ[:l_s])

    Keff_p_s_interface = Keff_i_j_interface(β_p_s, Keff_p, Keff_s)

    append!(Keff_p_medio, Keff_p_s_interface)

    ## Separator mean conductivity
    # Compute the harmonic mean values for the separator
    β_s = 0.5
    Keff_s_medio = Keff_i_medio(β_s, Keff_s)

    # The last element of Keff_s_medio will be the harmonic mean of the
    # elements at the interface separator-negative

    β_s_n = β_i_j(Δx.s, p.θ[:l_s], Δx.n, p.θ[:l_n])

    Keff_s_n_interface = Keff_i_j_interface(β_s_n, Keff_s, Keff_n)

    append!(Keff_s_medio, Keff_s_n_interface)

    ## Negative electrode mean conductivity
    # Compute the harmonic mean values for the negative electrode

    β_n = 0.5
    Keff_n_medio = Keff_i_medio(β_n, Keff_n)

    append!(Keff_n_medio, 0.0) # the cc interface is not used. The zero is only to match the dimensions

    return Keff_p_medio, Keff_s_medio, Keff_n_medio

end

harmonic_mean(β, x₁, x₂) = @. x₁*x₂/(β*x₂ + (1.0 - β)*x₁)

function interpolate_electrolyte_concentration(c_e, p::AbstractModel)
    """
    interpolate_electrolyte_concentration interpolates the value of electrolyte concentration at the edges of control volumes using harmonic mean
    """

    Δx = Δx_values(p.N)

    ## Electrolyte concentration interpolation

    # Interpolation within the positive electrode
    β_c_e_p = 0.5
    @inbounds @views c̄_e_p = harmonic_mean(β_c_e_p, c_e[1:p.N.p-1], c_e[2:p.N.p])

    # Interpolation on the interface between separator & positive electrode
    @inbounds @views β_c_e_ps = Δx.p*p.θ[:l_p]/2 / (Δx.p*p.θ[:l_p]/2 + Δx.s*p.θ[:l_s]/2)
    @inbounds @views c̄_e_ps = harmonic_mean(β_c_e_ps, c_e[p.N.p], c_e[p.N.p+1])

    # Interpolation within the separator
    β_c_e_s = 0.5
    @inbounds @views c̄_e_s = harmonic_mean(β_c_e_s, c_e[p.N.p+1:p.N.p+p.N.s-1], c_e[p.N.p+2:p.N.p+p.N.s])

    # Interpolation on the interface between separator & negative electrode
    @inbounds @views β_c_e_sn = Δx.s*p.θ[:l_s]/2 / (Δx.n*p.θ[:l_n]/2 + Δx.s*p.θ[:l_s]/2)
    @inbounds @views c̄_e_sn = harmonic_mean(β_c_e_sn, c_e[p.N.p+p.N.s], c_e[p.N.p+p.N.s+1])

    # Interpolation within the negative electrode
    β_c_e_n = 0.5
    @inbounds @views c̄_e_n = harmonic_mean(β_c_e_n, c_e[p.N.p+p.N.s+1:end-1], c_e[p.N.p+p.N.s+2:end])

    return [c̄_e_p; c̄_e_ps], [c̄_e_s; c̄_e_sn], c̄_e_n

end

interpolate_temperature(T, p::AbstractModel) = interpolate_electrolyte_concentration((@views @inbounds T[p.N.a+1:end-p.N.z]),p)

function interpolate_electrolyte_concetration_fluxes(c_e, p::AbstractModel)
    """
    interpolate_electrolyte_concetration_fluxes interpolates the electrolyte concentration flux at the edges of control volumes using harmonic mean
    """
    Δx = Δx_values(p.N)

    # Fluxes within the positive electrode
    @inbounds @views ∂ₓc_e_p = (c_e[2:p.N.p] .- c_e[1:p.N.p-1])/(Δx.p*p.θ[:l_p])

    # Fluxes at the separator-positive interface
    @inbounds @views ∂ₓc_e_ps = (c_e[p.N.p+1] .- c_e[p.N.p]) / ((Δx.p*p.θ[:l_p]/2+Δx.s*p.θ[:l_s]/2))

    # Fluxes within the separator
    @inbounds @views ∂ₓc_e_s = (c_e[p.N.p+2:p.N.p+p.N.s] .- c_e[p.N.p+1:p.N.p+p.N.s-1])/(Δx.s*p.θ[:l_s])

    # Fluxes at the separator-negative interface
    @inbounds @views ∂ₓc_e_sn = (c_e[p.N.p+p.N.s+1] .- c_e[p.N.p+p.N.s]) / ((Δx.n*p.θ[:l_n]/2+Δx.s*p.θ[:l_s]/2))

    # Fluxes within the negative electrode
    @inbounds @views ∂ₓc_e_n = (c_e[p.N.p+p.N.s+2:end] .- c_e[p.N.p+p.N.s+1:end-1])/(Δx.n*p.θ[:l_n])

    return [∂ₓc_e_p; ∂ₓc_e_ps], [∂ₓc_e_s; ∂ₓc_e_sn], ∂ₓc_e_n
end

Δx_values(N) = (p=1/N.p, s=1/N.s, n=1/N.n, a=1/N.a, z=1/N.z)