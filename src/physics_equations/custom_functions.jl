## Electrolyte functions
function D_s_eff_isothermal(c_s_avg_p, c_s_avg_n, T_p, T_n, p::AbstractParam)
    """
    D_s_eff_isothermal evaluates diffusion coefficients of the solid phase [m²/s].
    The user may modify the script to meet specific requirements
    """

    D_sp_eff = repeat([p.θ[:D_sp]], p.N.p)
    D_sn_eff = repeat([p.θ[:D_sn]], p.N.n)

    return D_sp_eff, D_sn_eff
end

function D_s_eff(c_s_avg_p, c_s_avg_n, T_p, T_n, p::AbstractParam)
    """
    D_s_eff evaluates diffusion coefficients of the solid phase [m²/s].
    The user may modify the script to meet specific requirements
    """

    R = 8.31446261815324

    T_ref = 298.15

    D_sp_eff = p.θ[:D_sp] .* exp.(-p.θ[:Ea_D_sp]/R*(1.0./T_p .- 1.0./T_ref))
    D_sn_eff = p.θ[:D_sn] .* exp.(-p.θ[:Ea_D_sn]/R*(1.0./T_n .- 1.0./T_ref))

    return D_sp_eff, D_sn_eff
end

function rxn_rate_isothermal(T_p, T_n, c_s_avg_p, c_s_avg_n, p::AbstractParam)
    """
    Reaction rates (k) of cathode and anode [m^2.5/(m^0.5 s)]
    """

    k_p = p.θ[:k_p]
    k_n = p.θ[:k_n]

    return k_p, k_n
end

function rxn_rate(T_p, T_n, c_s_avg_p, c_s_avg_n, p::AbstractParam)
    """
    Reaction rates (k) of cathode and anode [m^2.5/(m^0.5 s)]
    """
    
    R = 8.31446261815324
    T_ref = 298.15

    k_p = p.θ[:k_p] .* exp.(-(p.θ[:Ea_k_p]./R) .* (1.0./T_p .- 1.0./T_ref))
    k_n = p.θ[:k_n] .* exp.(-(p.θ[:Ea_k_n]./R) .* (1.0./T_n .- 1.0./T_ref))

    return k_p, k_n
end

function D_eff_linear(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    D_eff_linear evaluates the diffusion coefficients for the electrolyte phase [m^2/s]
    """

    D_eff_p = p.θ[:D_p].*p.θ[:ϵ_p].^p.θ[:brugg_p].*ones(p.N.p)
    D_eff_s = p.θ[:D_s].*p.θ[:ϵ_s].^p.θ[:brugg_s].*ones(p.N.s)
    D_eff_n = p.θ[:D_n].*p.θ[:ϵ_n].^p.θ[:brugg_n].*ones(p.N.n)
    
    return D_eff_p, D_eff_s, D_eff_n
end

function D_eff(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    D_eff evaluates the diffusion coefficients for the electrolyte phase [m^2/s]
    """
    
    func(c_e, T_p) = 1e-4.*10.0.^((-4.43 .- 54.0 ./ (T .- 229-5e-3 .* c_e) .-0.22e.-3.0.*c_e))

    D_eff_p = p.θ[:ϵ_p].^p.θ[:brugg_p].*func(c_e_p, T_p)
    D_eff_s = p.θ[:ϵ_s].^p.θ[:brugg_s].*func(c_e_s, T_s)
    D_eff_n = p.θ[:ϵ_n].^p.θ[:brugg_n].*func(c_e_n, T_n)

    return D_eff_p, D_eff_s, D_eff_n
end

function K_eff(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    K_eff evaluates the conductivity coefficients for the electrolyte phase [S/m]
    """

    func(c_e, T) = @. 1e-4*c_e*((-10.5 + 0.668*1e-3*c_e + 0.494*1e-6*c_e^2) + (0.074 - 1.78*1e-5*c_e - 8.86*1e-10*c_e^2)*T + (-6.96*1e-5 + 2.8*1e-8*c_e)*T^2)^2

    K_eff_p = p.θ[:ϵ_p].^p.θ[:brugg_p] .* func(c_e_p, T_p)
 K_eff_s = p.θ[:ϵ_s].^p.θ[:brugg_s] .* func(c_e_s, T_s)
 K_eff_n = p.θ[:ϵ_n].^p.θ[:brugg_n] .* func(c_e_n, T_n)

    return K_eff_p, K_eff_s, K_eff_n
end

function K_eff_isothermal(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    K_eff_isothermal evaluates the conductivity coefficients for the electrolyte phase [S/m]
    """

    func(c_e) = @. 4.1253*1e-2 + 5.007*1e-4*c_e - 4.7212*1e-7*c_e^2 + 1.5094*1e-10*c_e^3 - 1.6018*1e-14*c_e^4

    K_eff_p = p.θ[:ϵ_p].^p.θ[:brugg_p] .* func(c_e_p)
    K_eff_s = p.θ[:ϵ_s].^p.θ[:brugg_s] .* func(c_e_s)
    K_eff_n = p.θ[:ϵ_n].^p.θ[:brugg_n] .* func(c_e_n)

    return K_eff_p, K_eff_s, K_eff_n
end


## OCV – Positive Electrodes
function OCV_LCO(θ_p, T, p)
    T_ref = 25 + 273.15

    # Define the OCV for the positive electrode
    @views U_p = (-4.656 .+ 88.669θ_p.^2 .- 401.119θ_p.^4 .+ 342.909θ_p.^6 .- 462.471θ_p.^8 .+ 433.434θ_p.^10)./(-1 .+ 18.933θ_p.^2 .- 79.532θ_p.^4 .+ 37.311θ_p.^6 .- 73.083θ_p.^8 .+ 95.96θ_p.^10)

    # Compute the variation of OCV with respect to temperature variations [V/K]
    ∂U∂T_p = -0.001(0.199521039 .- 0.928373822θ_p .+ 1.364550689000003θ_p.^2 .- 0.6115448939999998θ_p.^3)./(1 .- 5.661479886999997θ_p +11.47636191θ_p.^2 .- 9.82431213599998θ_p.^3 .+ 3.048755063θ_p.^4)

    # if T == T_ref exactly (which may be true for isothermal runs), don't calculate ∂U∂T
    @inbounds for i in eachindex(U_p)
        U_p[i] = IfElse.ifelse(
            T[i] == T_ref,
            U_p[i],
            (T[i] - T_ref) * ∂U∂T_p[i]
        )
    end

    return U_p, ∂U∂T_p
end

function OCV_NCA(θ_p, T, p,
    β = [0.0774097629515982,0.606272598498384,0.152968841654083,0.0545480766604342,0.733284939659251,0.000712822478859885,0.0278763702136114,0.689404278770887,0.0400863812358319,-0.0456558058983228,0.0343726021084569,0.0457388625061757,-10.9970735885527,1.41606501867480,5.00624302823111,-77.6245627826084,0.782004325318630,0.0189932918328861,14.5073075667181,-0.499999383101730,8.96359519348470]
    )
    # Gaussian fit to orange line of Fig 1a: "Interrelationship Between the Open Circuit Potential Curves in a Class of Ni-Rich Cathode Materials"

    U_p = zeros(eltype(θ_p), length(θ_p))
    @inbounds for i in 1:3:length(β)
        @views @inbounds a,b,c = β[i:i+2]
        
        if i == 1
            U_p = @. a*exp(-((θ_p-b)/c)^2)
        else
            U_p .+= @. a*exp(-((θ_p-b)/c)^2)
        end
    end

    ∂U∂T_p = zeros(eltype(θ_p), length(θ_p))

    return U_p, ∂U∂T_p
end

function OCV_NCA_rational_fit_to_error(θ_p, T, p,
    β = [4.315671296056881, 33.33127933811027, -139.0002145630135, 431.515536401216, -759.8511037172485, -327.80324224830474, 819.4915334833945, 825.578736739238, 8.48259802857383, -42.39712512606054, 192.6881934984163, -532.7793984204297, 624.295657411788, -585.7452660304009, 596.5848375864856]
    )
    # rational polynomial fit to the result of OCV_data - U_n

    p0, p1, p2, p3, p4, p5, p6, p7,
    q1, q2, q3, q4, q5, q6, q7 = β

    U_p = @. (p0+p1*θ_p^1+p2*θ_p^2+p3*θ_p^3+p4*θ_p^4+p5*θ_p^5+p6*θ_p^6+p7*θ_p^7)/
             (1.0+q1*θ_p^1+q2*θ_p^2+q3*θ_p^3+q4*θ_p^4+q5*θ_p^5+q6*θ_p^6+q7*θ_p^7)

    ∂U∂T_p = zeros(eltype(U_p), length(U_p))

    return U_p, ∂U∂T_p
end

function OCV_NCA_Gaussian(θ_p, T, p,
    β = [2.72293031664864,-5.54307613383667,3.44082265769983,1.68850387671202,6.77740896498439,0.635529604225854,2.84626731937334,0.352292950679865,0.400043949630189,0.739171789504757,0.128777875250793,0.749152942395964,-0.296621801474026,2.88812544803969,0.632441834454358,0.486572547374672,-0.179415546506689,0.00216490525937275,0.223688995207503,0.0788499984276102,0.257906722352058,0.306207298112798,0.140671079987113,0.0903693552818221]
    )
    
    a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, c1, c2, c3, c4, c5, c6, c7, c8 = β

    U_p = @. a1*exp(-((θ_p-b1)/c1)^2) + a2*exp(-((θ_p-b2)/c2)^2) + a3*exp(-((θ_p-b3)/c3)^2) + a4*exp(-((θ_p-b4)/c4)^2) + a5*exp(-((θ_p-b5)/c5)^2) + a6*exp(-((θ_p-b6)/c6)^2) + a7*exp(-((θ_p-b7)/c7)^2) + a8*exp(-((θ_p-b8)/c8)^2)

    ∂U∂T_p = zeros(eltype(U_p), length(U_p))

    return U_p, ∂U∂T_p
end

function OCV_NCA_Cogswell(θ_p, T, p)
    # Dan Cogswell fit to Samsung battery
    kB = 1.38064852
    T_abs = 298.15
    e = 1.602176634e-19
    F = 96485.3365
    R = 8.31446261815324

    F = 96485.3365
    R = 8.31446261815324
    # Define the OCV for the positive electrode
    U_p = @. (-(R*T_abs)/F*log(θ_p/(1-θ_p)) + 4.12178 - 0.2338θ_p - 1.24566θ_p^2 + 1.16769θ_p^3 - 0.20745θ_p^4)

    ∂U∂T_p = zeros(eltype(U_p), length(U_p))

    return U_p, ∂U∂T_p
end

function OCV_NCA_Tesla(θ_p, T, p)
    
    F = 96485.3365
    R = 8.31446261815324

    # FFT fit to orange line of Fig 1a: "Interrelationship Between the Open Circuit Potential Curves in a Class of Ni-Rich Cathode Materials"
    β_OCV = [3.0895554222840045, 0.9266291218360826, 0.8717520795547863, 3.077124938532524, 0.9274018520417847, -0.8431499046574688, -327.54356833328274, -0.0028312090635908608, 0.9553271958053887, -0.3092343765681441, 0.20745494866568717, 24.065276897525592, -5.505815345994361, 25.01341116282674, -36.78988957769396, 16.244823555808825, 2.9867395541538726, -0.0323637411803452]

    U_p = @. β_OCV[1]/(1+exp(((1.0-θ_p)-β_OCV[2])*β_OCV[3]*F/(R*298.15))) + β_OCV[4]/(1+exp(((1.0-θ_p)-β_OCV[5])*β_OCV[6]*F/(R*298.15))) + β_OCV[7]/(1+exp(((1.0-θ_p)-β_OCV[8])*β_OCV[9]*F/(R*298.15))) + β_OCV[10]/(1+exp(((1.0-θ_p)-β_OCV[11])*β_OCV[12]*F/(R*298.15))) + β_OCV[13]/(1+exp(((1.0-θ_p)-β_OCV[14])*β_OCV[15]*F/(R*298.15))) + β_OCV[16]/(1+exp(((1.0-θ_p)-β_OCV[17])*β_OCV[18]*F/(R*298.15)))

    ∂U∂T_p = zeros(eltype(U_p), length(U_p))

    return U_p, ∂U∂T_p
end

## OCV – Negative Electrodes
function OCV_LiC6(θ_n, T, p)

    T_ref = 25 + 273.15

    # Calculate the open circuit voltage of the battery in the negative electrode
    U_n = @. 0.7222 + 0.1387*θ_n + 0.029*θ_n.^0.5 - 0.0172./θ_n + 0.0019./θ_n.^1.5 + 0.2808*exp(0.9-15*θ_n)-0.7984*exp(0.4465*θ_n - 0.4108)
 
    # Compute the variation of OCV with respect to temperature variations [V/K]
    ∂U∂T_n = @. 0.001*(0.005269056 +3.299265709*θ_n-91.79325798*θ_n.^2+1004.911008*θ_n.^3-5812.278127*θ_n.^4 + 19329.7549*θ_n.^5 - 37147.8947*θ_n.^6 + 38379.18127*θ_n.^7-16515.05308*θ_n.^8)
    ∂U∂T_n ./= @. (1-48.09287227*θ_n+1017.234804*θ_n.^2-10481.80419*θ_n.^3+59431.3*θ_n.^4-195881.6488*θ_n.^5 + 374577.3152*θ_n.^6 - 385821.1607*θ_n.^7 + 165705.8597*θ_n.^8)
    
    @inbounds for i in eachindex(U_n)
        U_n[i] = IfElse.ifelse(
            T[i] == T_ref,
            U_n[i],
            (T[i] - T_ref) * ∂U∂T_n[i]
        )
    end
    
    return U_n, ∂U∂T_n
end

function OCV_SiC(θ_n, T, p)
    # Calculate the open circuit voltage of the battery in the negative electrode
    U_n = @. 1.9793exp(-39.3631θ_n)+0.2482-0.0909tanh(29.8538(θ_n-0.1234))-0.04478tanh(14.9159(θ_n-0.2769))-0.0205tanh(30.4444(θ_n-0.6103))

    ∂U∂T_n = zeros(eltype(U_n), length(U_n))

    return U_n, ∂U∂T_n
end






## Reaction Rates
function rxn_BV(c_s_star, c_e, T, η, k_i, λ_MHC, c_s_max, p)
    # Butler-Volmer rate equation

    α = 0.5
    F = 96485.3365
    R = 8.31446261815324

    # simplified version which is faster to calculate
    if eltype(c_s_star) == Num

        if α == 0.5
            j_i = @. 2.0k_i*sqrt(c_e*c_s_star*(c_s_max - c_s_star))*(sinh(0.5*F*η/(R*T)))
        else
            j_i = @. k_i*(c_e)^(1 - α)*c_s_star^α*(c_s_max - c_s_star)^(1 - α)*(exp(α*F*η/(R*T)) - exp(-(α*F*η/(R*T))))
        end

    else # non-simplified form
        # normalizing the concentrations
        θ_i = c_s_star./c_s_max
        ĉ_e = c_e./p.θ[:c_e₀]
        η̂ = η.*(F./(R.*T))
        act_R = θ_i./(1.0 .- θ_i)

        γ_ts = 1.0./(1.0 .- θ_i) # needed for ecd_extras

        ecd = @. k_i*ĉ_e^(1.0 - α)*act_R^(α)/γ_ts
        j_i = @. ecd*(exp(-α*η̂) - exp((1.0 - α)*η̂))

        j_i .*= (-p.θ[:c_e₀].^(1.0 .- α).*c_s_max)

    end

    return j_i

end

function MHC_kfunc(η, λ)

    a = 1.0 + sqrt(λ)

    k = @. sqrt(π * λ) * (1.0 - erf((λ - sqrt((a) + η ^ 2)) / (2.0 * sqrt(λ)))) / (1.0 + exp(-η))

    return k
end

function rxn_MHC(c_s_star, c_e, T, η, k_i, λ_MHC, c_s_max, p)
    # MHC rate equation
    # See Zeng, Smith, Bai, Bazant 2014
    # Convert to "MHC overpotential"

    # normalizing the concentrations
    F = 96485.3365
    R = 8.31446261815324

    η̂ = η.*(F./(R.*T))
    θ_i = c_s_star./c_s_max
    ĉ_e = c_e./p.θ[:c_e₀]
    α = 0.5

    k₀ = k_i ./ MHC_kfunc(0.0, λ_MHC)

    η_f = η̂ .+ log.(ĉ_e./θ_i)

    if true # α == 0.5 && eltype(c_s_star) == Num

        a = 1.0 + sqrt(λ_MHC)

        k₀ = @. k_i / ((1.0 - erf((λ_MHC - sqrt((a))) / (2.0 * sqrt(λ_MHC)))) / 2.0)
        
        coeff_rd_ox = @. k₀*((1.0 - erf((λ_MHC - sqrt((a) + η_f ^ 2)) / (2.0 * sqrt(λ_MHC)))))

        j_i = @. coeff_rd_ox * (1.0/(1.0 + exp(-η_f))*p.θ[:c_e₀]*c_s_star - 1.0/(1.0 + exp(+η_f))*c_e*c_s_max) * sqrt((1.0 - c_s_star/c_s_max)/p.θ[:c_e₀])
    elseif false

        k_rd = k₀.*MHC_kfunc(-η_f, λ_MHC)
        k_ox = k₀.*MHC_kfunc(+η_f, λ_MHC)

        j_i = @. (k_ox*p.θ[:c_e₀]*c_s_star - k_rd*c_e*c_s_max)*sqrt((c_s_max - c_s_star)/(p.θ[:c_e₀] * c_s_max))
    else

        k_rd = k₀.*MHC_kfunc(-η_f, λ_MHC)
        k_ox = k₀.*MHC_kfunc(+η_f, λ_MHC)

        act_R = θ_i./(1.0 .- θ_i)

        γ_ts = 1.0./(1.0 .- θ_i) # needed for ecd_extras

        ecd_extras = ĉ_e.^(1.0 .- α) .* act_R.^α ./ (γ_ts.*sqrt.(ĉ_e.*θ_i))


        j_i = ecd_extras.*(k_rd.*ĉ_e .- k_ox.*θ_i)
        j_i .*= (-p.θ[:c_e₀].^(1.0 .- α).*c_s_max) # need this to match LIONSIMBA dimensional eqn.

        j_i = @. (c_s_max - c_s_star)*((p.θ[:c_e₀]*c_s_star*k_ox - c_e*c_s_max*k_rd)*((c_s_star*(c_s_max - c_s_star))/(c_s_max*c_e*p.θ[:c_e₀]))^(1/2))
    end

    return j_i
end

function rxn_BV_γMod_01(c_s_star, c_e, T, η, k_i, λ_MHC, c_s_max, p)
    # Butler-Volmer rate equation

    # normalizing the concentrations
    F = 96485.3365
    R = 8.31446261815324
    θ_i = c_s_star./c_s_max
    ĉ_e = c_e./p.θ[:c_e₀]
    η̂ = η.*(F./(R.*T))
    act_R = θ_i./(1.0 .- θ_i)
    α = 0.5

    γ_ts = 1.0./(θ_i.*(1.0 .- θ_i)) # needed for ecd_extras

    ecd = @. k_i * ĉ_e^(1-α) * act_R^(α) / γ_ts
    j_i = @. ecd * (exp(-α*η̂) - exp((1.0-α)*η̂))

    j_i .*= (-p.θ[:c_e₀].^(1.0 .- α).*c_s_max)

    return j_i

end
