function firstOrderDerivativeMatrix(xl, xu, n)
    Δx    = (xu - xl)/(n - 1)
    r8fΔx = 1/(40320*Δx)
    nm4   = n - 4

    mid_block = zeros(Float64, n-8, n)

    first_row 	= [-109584.0 	+322560 	-564480 	+752640 	-705600 	+451584 	-188160 	+46080 	-5040]
    second_row 	= [-5040.0 	-64224 		+141120 	-141120 	+117600 	-70560 		+28224 		-6720 	+720]
    third_row 	= [+720.0 	-11520 		-38304 		+80640 		-50400 		+26880 		-10080 		+2304 	-240]
    fourth_row 	= [-240.0 	+2880 		-20160 		-18144 		+50400 		-20160 		+6720 		-1440 	+144]

    i_th_row 	= [+144.0 	-1536 		+8064 		-32256 		+0 			+32256 		-8064 		+1536 	-144]

    n_min_3_row 	= [-144.0 	+1440 		-6720 		+20160 		-50400 		+18144 		+20160 		-2880 	+240]
    n_min_2_row 	= [+240.0 	-2304 		+10080 		-26880 		+50400 		-80640 		+38304 		+11520 	-720]
    n_min_1_row 	= [-720.0 	+6720 		-28224 		+70560 		-117600 	+141120 	-141120 	+64224 	+5040]
    n_min_0_row 	= [+5040.0 	-46080 		+188160  	-451584 	+705600 	-752640 	+564480 	-322560 +109584]

    first_block = [first_row; second_row; third_row; fourth_row]
    first_block = [first_block zeros(Float64, 4, n-9)]

    @inbounds for (row_index, i) in enumerate(5:nm4)
        @inbounds mid_block[row_index,row_index:row_index+8] = i_th_row
    end

    last_block = [n_min_3_row; n_min_2_row; n_min_1_row; n_min_0_row]
    last_block=[zeros(Float64, 4, n-9) last_block]

    derivativeMatrix = [first_block;mid_block;last_block]

    return derivativeMatrix, r8fΔx
end

function secondOrderDerivativeMatrix(xl, xu, n)
    Δx 		= (xu - xl)/(n - 1)

    r12Δxs 	= 1/(12*Δx^2)

    mid_block = zeros(Float64, n - 4, n)

    first_row       = [-415/6   +96     -36     +32/3       -3/2    0]
    second_row      = [+10.0      -15     -4      +14         -6      +1]
    i_th_row        = [-1.0       +16     -30     +16         -1]
    semi_last_row   = [+1.0       -6      +14     -4          -15     +10]
    last_row        = [0.0   -3/2    +32/3   -36     +96     -415/6]

    first_block = [first_row; second_row]
    first_block = [first_block zeros(Float64, 2, n-6)]


    last_block 	= [semi_last_row; last_row]
    last_block 	= [zeros(Float64, 2, n-6) last_block]

    @inbounds for (row_index, i) in enumerate(3:n-2)
        @inbounds mid_block[row_index, row_index:row_index+4] = i_th_row
    end

    derivativeMatrix = 6*[first_block; mid_block; last_block]

    return derivativeMatrix, r12Δxs, Δx
end

function DerivativeMatrices(N)
    @inbounds FO_D_i, FO_D_c_i            = firstOrderDerivativeMatrix( 0.0, 1.0, N)
    @inbounds SO_D_i, SO_D_c_i, SO_D_Δx_i = secondOrderDerivativeMatrix(0.0, 1.0, N)

    return FO_D_i, FO_D_c_i, SO_D_i, SO_D_c_i, SO_D_Δx_i
end

function blockMatrixMaker(p, X, Y, Z)
    A_tot = zeros(eltype(X), (p.N.p+p.N.s+p.N.n), (p.N.p+p.N.s+p.N.n))

    ind_0diag    = diagind(A_tot)
    ind_neg1diag = diagind(A_tot, -1)
    ind_pos1diag = diagind(A_tot, 1)

    # p
    @inbounds @views A_tot[ind_0diag[1:p.N.p]]      = X
    @inbounds @views A_tot[ind_0diag[2:p.N.p]]     += X[1:end-1]
    @inbounds @views A_tot[ind_neg1diag[1:p.N.p-1]] = -X[1:end-1]
    @inbounds @views A_tot[ind_pos1diag[1:p.N.p-1]] = -X[1:end-1]

    # s
    @inbounds @views A_tot[ind_0diag[(1:p.N.s) .+ p.N.p]]      = Y
    @inbounds @views A_tot[ind_0diag[(2:p.N.s) .+ p.N.p]]     += Y[1:end-1]
    @inbounds @views A_tot[ind_neg1diag[(1:p.N.s-1) .+ p.N.p]] = -Y[1:end-1]
    @inbounds @views A_tot[ind_pos1diag[(1:p.N.s-1) .+ p.N.p]] = -Y[1:end-1]
    # n
    @inbounds @views A_tot[ind_0diag[(1:p.N.n) .+ (p.N.p + p.N.s)]]      = Z
    @inbounds @views A_tot[ind_0diag[(2:p.N.n) .+ (p.N.p + p.N.s)]]     += Z[1:end-1]
    @inbounds @views A_tot[ind_neg1diag[(1:p.N.n-1) .+ (p.N.p + p.N.s)]] = -Z[1:end-1]
    @inbounds @views A_tot[ind_pos1diag[(1:p.N.n-1) .+ (p.N.p + p.N.s)]] = -Z[1:end-1]

    return A_tot
end

function interpolateElectrolyteConductivities(Keff_p, Keff_s, Keff_n, p)
    #	interpolateElectrolyteConductivities interpolates electrolyte conductivities at the edges of control volumes using harmonic mean.

    Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)

    @views @inbounds Keff_i_medio(β_i, Keff_i)                 = Keff_i[1:end-1].*Keff_i[2:end]./(β_i*Keff_i[2:end]+(1-β_i)*Keff_i[1:end-1])
    @views @inbounds β_i_j(Δx_i, l_i, Δx_j, l_j)               = Δx_i*l_i/2 /(Δx_j*l_j/2+Δx_i*l_i/2)
    @views @inbounds Keff_i_j_interface(β_i_j, Keff_i, Keff_j) = Keff_i[end]*Keff_j[1] / (β_i_j*Keff_j[1] + (1-β_i_j)*Keff_i[end])

    ## Positive electrode mean conductivity
    β_p = 0.5
    @views @inbounds Keff_p_medio = Keff_i_medio(β_p, Keff_p)

    # The last element of Keff_p_medio will be the harmonic mean of the
    # elements at the interface positive-separator

    @views @inbounds β_p_s = β_i_j(Δx_p, p.θ[:l_p], Δx_s, p.θ[:l_s])

    Keff_p_s_interface = Keff_i_j_interface(β_p_s, Keff_p, Keff_s)

    @inbounds append!(Keff_p_medio, Keff_p_s_interface)

    ## Separator mean conductivity
    # Compute the harmonic mean values for the separator
    β_s = 0.5
    @views @inbounds Keff_s_medio = Keff_i_medio(β_s, Keff_s)

    # The last element of Keff_s_medio will be the harmonic mean of the
    # elements at the interface separator-negative

    @views @inbounds β_s_n = β_i_j(Δx_s, p.θ[:l_s], Δx_n, p.θ[:l_n])

    Keff_s_n_interface = Keff_i_j_interface(β_s_n, Keff_s, Keff_n)

    @inbounds append!(Keff_s_medio, Keff_s_n_interface)

    ## Negative electrode mean conductivity
    # Compute the harmonic mean values for the negative electrode

    β_n = 0.5
    @views @inbounds Keff_n_medio = Keff_i_medio(β_n, Keff_n)

    @inbounds append!(Keff_n_medio, 0.0) # the cc interface is not used. The zero is only to match the dimensions

    return Keff_p_medio, Keff_s_medio, Keff_n_medio

end

function harmonic_mean(β, x₁, x₂)
    x₁.*x₂ ./ (β*x₂ + (1.0 - β)*x₁)
end

function interpolateElectrolyteConcentration(c_e, p)
    #	interpolateElectrolyteConcentration interpolates the value of electrolyte concentration at the edges of control volumes using harmonic mean.

    Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)

    ## Electrolyte concentration interpolation

    # Interpolation within the positive electrode
    β_c_e_p = 0.5
    @inbounds @views c̄_e_p = harmonic_mean(β_c_e_p, c_e[1:p.N.p-1], c_e[2:p.N.p])

    # Interpolation on the interface between separator & positive electrode
    @inbounds @views β_c_e_ps = Δx_p*p.θ[:l_p]/2 / (Δx_p*p.θ[:l_p]/2 + Δx_s*p.θ[:l_s]/2)
    @inbounds @views c̄_e_ps = harmonic_mean(β_c_e_ps, c_e[p.N.p], c_e[p.N.p+1])

    # Interpolation within the separator
    β_c_e_s = 0.5
    @inbounds @views c̄_e_s = harmonic_mean(β_c_e_s, c_e[p.N.p+1:p.N.p+p.N.s-1], c_e[p.N.p+2:p.N.p+p.N.s])

    # Interpolation on the interface between separator & negative electrode
    @inbounds @views β_c_e_sn = Δx_s*p.θ[:l_s]/2 / (Δx_n*p.θ[:l_n]/2 + Δx_s*p.θ[:l_s]/2)
    @inbounds @views c̄_e_sn = harmonic_mean(β_c_e_sn, c_e[p.N.p+p.N.s], c_e[p.N.p+p.N.s+1])

    # Interpolation within the negative electrode
    β_c_e_n = 0.5
    @inbounds @views c̄_e_n = harmonic_mean(β_c_e_n, c_e[p.N.p+p.N.s+1:end-1], c_e[p.N.p+p.N.s+2:end])

    return c̄_e_p, c̄_e_ps, c̄_e_s, c̄_e_sn, c̄_e_n

end

function interpolateTemperature(T, p)
    #	interpolateTemperature evaluates the interpolation of the temperature at the edges of control volumes using harmonic mean.

    Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)

    # Interpolation within the positive electrode
    β_T_p = 0.5
    @inbounds @views T̄_p = harmonic_mean(β_T_p, T[p.N.a+1:p.N.a+p.N.p-1], T[p.N.a+2:p.N.a+p.N.p])

    # Interpolation on the interface between separator & positive electrode
    @inbounds @views β_T_ps = Δx_p*p.θ[:l_p]/2 / (Δx_p*p.θ[:l_p]/2 + Δx_s*p.θ[:l_s]/2)
    @inbounds @views T̄_ps = harmonic_mean(β_T_ps, T[p.N.a+p.N.p], T[p.N.a+p.N.p+1])

    # Interpolation within the separator
    β_T_s = 0.5
    @inbounds @views T̄_s = harmonic_mean(β_T_s, T[p.N.a+p.N.p+1:p.N.a+p.N.p+p.N.s-1], T[p.N.a+p.N.p+2:p.N.a+p.N.p+p.N.s])

    # Interpolation on the interface between separator & negative electrode
    @inbounds @views β_T_sn = Δx_s*p.θ[:l_s]/2 / (Δx_n*p.θ[:l_n]/2 + Δx_s*p.θ[:l_s]/2)
    @inbounds @views T̄_sn = harmonic_mean(β_T_sn, T[p.N.a+p.N.p+p.N.s], T[p.N.a+p.N.p+p.N.s+1])

    # Interpolation within the negative electrode
    β_T_n = 0.5
    @inbounds @views T̄_n = harmonic_mean(β_T_n, T[p.N.a+p.N.p+p.N.s+1:end-(p.N.z)-1], T[p.N.a+p.N.p+p.N.s+2:end-p.N.z])

return T̄_p, T̄_ps, T̄_s, T̄_sn, T̄_n

end


function interpolateElectrolyteConcetrationFluxes(c_e, p)
    #	interpolateElectrolyteConcetrationFluxes interpolates the electrolyte concentration flux at the edges of control volumes using harmonic mean.

    Δx_p, Δx_s, Δx_n, Δx_a, Δx_z = Δx(p)

    # Fluxes within the positive electrode
    @inbounds @views c_e_flux_p = (c_e[2:p.N.p] .- c_e[1:p.N.p-1])/(Δx_p*p.θ[:l_p])

    # Fluxes at the separator-positive interface
    @inbounds @views c_e_flux_ps = (c_e[p.N.p+1] .- c_e[p.N.p]) / ((Δx_p*p.θ[:l_p]/2+Δx_s*p.θ[:l_s]/2))

    # Fluxes within the separator
    @inbounds @views c_e_flux_s = (c_e[p.N.p+2:p.N.p+p.N.s] .- c_e[p.N.p+1:p.N.p+p.N.s-1])/(Δx_s*p.θ[:l_s])

    # Fluxes at the separator-negative interface
    @inbounds @views c_e_flux_sn = (c_e[p.N.p+p.N.s+1] .- c_e[p.N.p+p.N.s]) / ((Δx_n*p.θ[:l_n]/2+Δx_s*p.θ[:l_s]/2))

    # Fluxes within the negative electrode
    @inbounds @views c_e_flux_n = (c_e[p.N.p+p.N.s+2:end] .- c_e[p.N.p+p.N.s+1:end-1])/(Δx_n*p.θ[:l_n])

    return c_e_flux_p, c_e_flux_ps, c_e_flux_s, c_e_flux_sn, c_e_flux_n

end

function Δx(p)

    Δx_p = 1.0/p.N.p
    Δx_s = 1.0/p.N.s
    Δx_n = 1.0/p.N.n
    Δx_a = 1.0/p.N.a
    Δx_z = 1.0/p.N.z

    return Δx_p, Δx_s, Δx_n, Δx_a, Δx_z

end