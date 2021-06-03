temperature_switch(a,b,c,p::AbstractParam) = p.numerics.temperature === false ? IfElse.ifelse.(a,b,c) : c

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

    R = const_Ideal_Gas

    T_ref = 298.15

    # oftentimes T == T_ref for isothermal experiments. the if statement prevents unnecessary calculations
    D_sp_eff = p.θ[:D_sp] .* temperature_switch( T_p .== T_ref, 1, exp.(-p.θ[:Ea_D_sp]/R*(1.0./T_p .- 1.0./T_ref)), p )
    D_sn_eff = p.θ[:D_sn] .* temperature_switch( T_n .== T_ref, 1, exp.(-p.θ[:Ea_D_sn]/R*(1.0./T_n .- 1.0./T_ref)), p )

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
    
    R = const_Ideal_Gas
    T_ref = 298.15

    # oftentimes T == T_ref for isothermal experiments. the if statement prevents unnecessary calculations
    k_p = p.θ[:k_p] .* temperature_switch( T_p .== T_ref, 1, exp.(-(p.θ[:Ea_k_p]./R) .* (1.0./T_p .- 1.0./T_ref)), p )
    k_n = p.θ[:k_n] .* temperature_switch( T_n .== T_ref, 1, exp.(-(p.θ[:Ea_k_n]./R) .* (1.0./T_n .- 1.0./T_ref)), p )

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

function D_eff_linear_one_term(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    D_eff_linear evaluates the diffusion coefficients for the electrolyte phase [m^2/s]
    """

    D_eff_p = p.θ[:D_e].*p.θ[:ϵ_p].^p.θ[:brugg_p].*ones(p.N.p)
    D_eff_s = p.θ[:D_e].*p.θ[:ϵ_s].^p.θ[:brugg_s].*ones(p.N.s)
    D_eff_n = p.θ[:D_e].*p.θ[:ϵ_n].^p.θ[:brugg_n].*ones(p.N.n)
    
    return D_eff_p, D_eff_s, D_eff_n
end

function D_eff(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    D_eff evaluates the diffusion coefficients for the electrolyte phase [m^2/s]
    """
    
    func(c_e, T) = 1e-4*10.0^((-4.43 - 54.0 / (T - 229-5e-3 * c_e) - 0.22e-3 * c_e))

    D_eff_p = p.θ[:ϵ_p].^p.θ[:brugg_p].*func.(c_e_p, T_p)
    D_eff_s = p.θ[:ϵ_s].^p.θ[:brugg_s].*func.(c_e_s, T_s)
    D_eff_n = p.θ[:ϵ_n].^p.θ[:brugg_n].*func.(c_e_n, T_n)

    return D_eff_p, D_eff_s, D_eff_n
end

function D_eff_EC_EMC_DMC(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    D_eff evaluates the diffusion coefficients for the electrolyte phase [m^2/s]
    """
    
    a00,a01,a11,a21,a31,a41,a02,a12,a22,a32,a42 = 
        (1.8636977199162228e-8, -1.3917476882039536e-10, 3.1325506789441764e-14,
        -7.300511963906146e-17, 5.119530992181613e-20, -1.1514201913496038e-23,
        2.632651793626908e-13, -1.1262923552112963e-16, 2.614674626287283e-19,
        -1.8321158900930459e-22, 4.110643579807774e-26)
    
    func(c_e, T=298.15) = a00 + a01*T + a11*c_e*T + a21*c_e^2*T + a31*c_e^3*T +
        a41*c_e^4*T + a02*T^2 + a12*c_e*T^2 + a22*c_e^2*T^2 + a32*c_e^3*T^2 + a42*c_e^4*T^2

    D_eff_p = p.θ[:ϵ_p].^p.θ[:brugg_p].*func.(c_e_p, T_p)
    D_eff_s = p.θ[:ϵ_s].^p.θ[:brugg_s].*func.(c_e_s, T_s)
    D_eff_n = p.θ[:ϵ_n].^p.θ[:brugg_n].*func.(c_e_n, T_n)

    return D_eff_p, D_eff_s, D_eff_n
end

function K_eff_EC_EMC_DMC(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    K_eff evaluates the conductivity coefficients for the electrolyte phase [S/m]
    """
    
    func(c_e, T) = -0.5182386306736273 +
        - 0.0065182740160006376 * c_e +
        + 0.0016958426698238335 * T +
        + 1.4464586693911396e-6 * c_e^2 +
        + 3.0336049598190174e-5 * c_e*T +
        + 3.046769609846814e-10 * c_e^3 +
        - 1.0493995729897995e-8 * c_e^2*T

    K_eff_p = p.θ[:ϵ_p].^p.θ[:brugg_p] .* func.(c_e_p, T_p)
    K_eff_s = p.θ[:ϵ_s].^p.θ[:brugg_s] .* func.(c_e_s, T_s)
    K_eff_n = p.θ[:ϵ_n].^p.θ[:brugg_n] .* func.(c_e_n, T_n)
    
    return K_eff_p, K_eff_s, K_eff_n
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
function OCV_LCO(θ_p, T=298.15, p=nothing)
    T_ref = 25 + 273.15

    # Define the OCV for the positive electrode
    @views U_p = (-4.656 .+ 88.669θ_p.^2 .- 401.119θ_p.^4 .+ 342.909θ_p.^6 .- 462.471θ_p.^8 .+ 433.434θ_p.^10)./(-1 .+ 18.933θ_p.^2 .- 79.532θ_p.^4 .+ 37.311θ_p.^6 .- 73.083θ_p.^8 .+ 95.96θ_p.^10)

    # Compute the variation of OCV with respect to temperature variations [V/K]
    ∂U∂T_p = -0.001(0.199521039 .- 0.928373822θ_p .+ 1.364550689000003θ_p.^2 .- 0.6115448939999998θ_p.^3)./(1 .- 5.661479886999997θ_p +11.47636191θ_p.^2 .- 9.82431213599998θ_p.^3 .+ 3.048755063θ_p.^4)

    # if T == T_ref exactly (which may be true for isothermal runs), don't calculate ∂U∂T
    U_p += temperature_switch( T .== T_ref, 0, ∂U∂T_p.*(T .- T_ref), p )

    return U_p, ∂U∂T_p
end

function OCV_NCA(x, T=298.15, p=nothing)
    
    a0,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7,w = 
        (11176089.4030491,-57362910.6578415,-37181629.3551737,64732409.0190326,46282329.2204589,-34194136.7536203,-28342089.2834402,7232227.74453184,9650246.65198595,1243026.08199635,-1697305.24472650,-936104.430657431,112362.897540026,136155.099295619,0.897597901025655)

    lb,ub=(0.1667, 0.8463)
    θ_p = @. ((x-lb)/(ub-lb))*(0.8643057647313726 - -0.0013051813260597826) + -0.0013051813260597826
        

    U_p = @. a0+b1*sin(1*θ_p*w)+a2*cos(2*θ_p*w)+b2*sin(2*θ_p*w)+a3*cos(3*θ_p*w)+b3*sin(3*θ_p*w)+a4*cos(4*θ_p*w)+b4*sin(4*θ_p*w)+a5*cos(5*θ_p*w)+b5*sin(5*θ_p*w)+a6*cos(6*θ_p*w)+b6*sin(6*θ_p*w)+a7*cos(7*θ_p*w)+b7*sin(7*θ_p*w)

    ∂U∂T_p = zeros(eltype(U_p), length(U_p))

    return U_p, ∂U∂T_p
end

function OCV_NCA_rational_fit_to_error(θ_p, T=298.15, p=nothing,
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

function OCV_NCA_Gaussian(θ_p, T=298.15, p=nothing,
    β = [2.72293031664864,-5.54307613383667,3.44082265769983,1.68850387671202,6.77740896498439,0.635529604225854,2.84626731937334,0.352292950679865,0.400043949630189,0.739171789504757,0.128777875250793,0.749152942395964,-0.296621801474026,2.88812544803969,0.632441834454358,0.486572547374672,-0.179415546506689,0.00216490525937275,0.223688995207503,0.0788499984276102,0.257906722352058,0.306207298112798,0.140671079987113,0.0903693552818221]
    )
    
    a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, c1, c2, c3, c4, c5, c6, c7, c8 = β

    U_p = @. a1*exp(-((θ_p-b1)/c1)^2) + a2*exp(-((θ_p-b2)/c2)^2) + a3*exp(-((θ_p-b3)/c3)^2) + a4*exp(-((θ_p-b4)/c4)^2) + a5*exp(-((θ_p-b5)/c5)^2) + a6*exp(-((θ_p-b6)/c6)^2) + a7*exp(-((θ_p-b7)/c7)^2) + a8*exp(-((θ_p-b8)/c8)^2)

    ∂U∂T_p = zeros(eltype(U_p), length(U_p))

    return U_p, ∂U∂T_p
end

function OCV_NCA_Cogswell(θ_p, T=298.15, p=nothing)
    # Dan Cogswell fit to Samsung battery
    kB = 1.38064852
    T_abs = 298.15
    e = 1.602176634e-19
    F = const_Faradays
    R = const_Ideal_Gas

    F = const_Faradays
    R = const_Ideal_Gas
    # Define the OCV for the positive electrode
    U_p = @. (-(R*T_abs)/F*log(θ_p/(1-θ_p)) + 4.12178 - 0.2338θ_p - 1.24566θ_p^2 + 1.16769θ_p^3 - 0.20745θ_p^4)

    ∂U∂T_p = zeros(eltype(U_p), length(U_p))

    return U_p, ∂U∂T_p
end

## OCV – Negative Electrodes
function OCV_LiC6(θ_n, T=298.15, p=nothing)
    T_ref = 25 + 273.15

    # Calculate the open circuit voltage of the battery in the negative electrode
    U_n = @. 0.7222 + 0.1387*θ_n + 0.029*sqrt_ReLU(θ_n) - 0.0172./θ_n + 0.0019./(sqrt_ReLU(θ_n;minval=1e-4).*θ_n) + 0.2808*exp(0.9-15*θ_n)-0.7984*exp(0.4465*θ_n - 0.4108)
 
    # Compute the variation of OCV with respect to temperature variations [V/K]
    ∂U∂T_n = @. 0.001*(0.005269056 +3.299265709*θ_n-91.79325798*θ_n.^2+1004.911008*θ_n.^3-5812.278127*θ_n.^4 + 19329.7549*θ_n.^5 - 37147.8947*θ_n.^6 + 38379.18127*θ_n.^7-16515.05308*θ_n.^8)
    ∂U∂T_n ./= @. (1-48.09287227*θ_n+1017.234804*θ_n.^2-10481.80419*θ_n.^3+59431.3*θ_n.^4-195881.6488*θ_n.^5 + 374577.3152*θ_n.^6 - 385821.1607*θ_n.^7 + 165705.8597*θ_n.^8)
    
    U_n += temperature_switch( T .== T_ref, 0, ∂U∂T_n.*(T .- T_ref), p )
    
    return U_n, ∂U∂T_n
end

function OCV_SiC(x, T=298.15, p=nothing)
    # Calculate the open circuit voltage of the battery in the negative electrode
    a, b, c, d, e, f, g, h, k, l, m, n, p, q, r = 
        (-48.9921992984694,29.9816001180044,21.1566495684493,-65.0631963216785,-0.00318535894916062,0.00107700919632965,-47.7685802868867,-17.5907349638410,6.53454664441369,161.854109570929,-0.103375178956369,0.0899080798075340,-0.283281555638378,0.0359795602165089,0.0393071531227136)
    

    lb,ub=(0.0150, 0.9896)
    θ_n = @. 1 - ( ((x-lb)/(ub-lb))*(0.08684185097542697 - 0.9913690894300142) + 0.9913690894300142)

    U_n = @. a+b*exp(-c*θ_n)+d*tanh((θ_n-e)/f)+g*tanh((θ_n-h)/k)+l*tanh((θ_n-m)/n)+p*tanh((θ_n-q)/r)

    ∂U∂T_n = zeros(eltype(U_n), length(U_n))

    return U_n, ∂U∂T_n
end






## Reaction Rates
sqrt_ReLU(x;minval=0) = sqrt(max(minval,x))

function rxn_BV(c_s_star, c_e, T, η, k_i, λ_MHC, c_s_max, p;
    sqrt=sqrt_ReLU
    )
    # The modified sqrt function is to avoid errors when concentrations momentarily become non-physical

    # Butler-Volmer rate equation

    α = 0.5
    F = const_Faradays
    R = const_Ideal_Gas

    # simplified version which is faster to calculate
    if true #eltype(c_s_star) == Num

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

function rxn_MHC(c_s_star, c_e, T, η, k_i, λ_MHC, c_s_max, p;
    sqrt=sqrt_ReLU
    )
    # The modified sqrt function is to avoid errors when concentrations momentarily become non-physical

    # MHC rate equation
    # See Zeng, Smith, Bai, Bazant 2014
    # Convert to "MHC overpotential"

    # normalizing the concentrations
    F = const_Faradays
    R = const_Ideal_Gas

    η̂ = η.*(F./(R.*T))
    θ_i = c_s_star./c_s_max
    ĉ_e = c_e./p.θ[:c_e₀]
    α = 0.5

    k₀ = k_i ./ MHC_kfunc(0.0, λ_MHC)

    log_ReLU(x;minval=0) = log(max(minval,x))
    η_f = η̂ .+ log_ReLU.(ĉ_e./θ_i;minval=1e-4)

    if α == 0.5

        a = 1.0 + sqrt(λ_MHC)

        k₀ = @. k_i / ((1.0 - erf((λ_MHC - sqrt((a))) / (2.0 * sqrt(λ_MHC)))) / 2.0)
        
        coeff_rd_ox = @. k₀*((1.0 - erf((λ_MHC - sqrt((a) + η_f ^ 2)) / (2.0 * sqrt(λ_MHC)))))

        j_i = @. coeff_rd_ox * (1.0/(1.0 + exp(-η_f))*p.θ[:c_e₀]*c_s_star - 1.0/(1.0 + exp(+η_f))*c_e*c_s_max) * sqrt_ReLU((1.0 - c_s_star/c_s_max)/p.θ[:c_e₀])
    elseif false

        k_rd = k₀.*MHC_kfunc(-η_f, λ_MHC)
        k_ox = k₀.*MHC_kfunc(+η_f, λ_MHC)

        j_i = @. (k_ox*p.θ[:c_e₀]*c_s_star - k_rd*c_e*c_s_max)*sqrt_ReLU((c_s_max - c_s_star)/(p.θ[:c_e₀] * c_s_max))
    else

        k_rd = k₀.*MHC_kfunc(-η_f, λ_MHC)
        k_ox = k₀.*MHC_kfunc(+η_f, λ_MHC)

        act_R = θ_i./(1.0 .- θ_i)

        γ_ts = 1.0./(1.0 .- θ_i) # needed for ecd_extras

        ecd_extras = ĉ_e.^(1.0 .- α) .* act_R.^α ./ (γ_ts.*sqrt.(ĉ_e.*θ_i))


        j_i = ecd_extras.*(k_rd.*ĉ_e .- k_ox.*θ_i)
        j_i .*= (-p.θ[:c_e₀].^(1.0 .- α).*c_s_max) # need this to match LIONSIMBA dimensional eqn.

        j_i = @. (c_s_max - c_s_star)*((p.θ[:c_e₀]*c_s_star*k_ox - c_e*c_s_max*k_rd)*sqrt_ReLU((c_s_star*(c_s_max - c_s_star))/(c_s_max*c_e*p.θ[:c_e₀])))
    end

    return j_i
end

function rxn_BV_γMod_01(c_s_star, c_e, T, η, k_i, λ_MHC, c_s_max, p)
    # Butler-Volmer rate equation

    # normalizing the concentrations
    F = const_Faradays
    R = const_Ideal_Gas
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
