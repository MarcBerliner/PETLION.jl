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

    # oftentimes T == T_ref for isothermal experiments, so the switch prevents unnecessary calculations
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
    U_p = (-4.656 .+ 88.669θ_p.^2 .- 401.119θ_p.^4 .+ 342.909θ_p.^6 .- 462.471θ_p.^8 .+ 433.434θ_p.^10)./(-1 .+ 18.933θ_p.^2 .- 79.532θ_p.^4 .+ 37.311θ_p.^6 .- 73.083θ_p.^8 .+ 95.96θ_p.^10)

    # Compute the variation of OCV with respect to temperature variations [V/K]
    ∂U∂T_p = -0.001(0.199521039 .- 0.928373822θ_p .+ 1.364550689000003θ_p.^2 .- 0.6115448939999998θ_p.^3)./(1 .- 5.661479886999997θ_p +11.47636191θ_p.^2 .- 9.82431213599998θ_p.^3 .+ 3.048755063θ_p.^4)

    # if T == T_ref exactly (which may be true for isothermal runs), don't calculate ∂U∂T
    U_p += temperature_switch( T .== T_ref, 0, ∂U∂T_p.*(T .- T_ref), p )

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

function OCV_NMC(θ_p, T=298.15, p=nothing)
    # Define the OCV for the positive electrode
    U_p = @. -10.72θ_p^4+ 23.88θ_p^3 - 16.77θ_p^2 + 2.595θ_p + 4.563

    # Compute the variation of OCV with respect to temperature variations [V/K]
    ∂U∂T_p = zeros(length(U_p))

    return U_p, ∂U∂T_p
end

function OCV_LiC6_with_NMC(θ_n, T=298.15, p=nothing)
    # Define the OCV for the negative electrode
    U_n = @. 0.1493 + 0.8493exp(-61.79θ_n) + 0.3824exp(-665.8θ_n) - 
        exp(39.42θ_n-41.92) - 0.03131atan(25.59θ_n - 4.099) - 
        0.009434atan(32.49θ_n - 15.74)

    # Compute the variation of OCV with respect to temperature variations [V/K]
    ∂U∂T_n = zeros(length(U_n))

    return U_n, ∂U∂T_n
end








## Thermodynamic factors
function thermodynamic_factor_linear(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    Thermodynamic factor for the activity coefficient. The term `(1-t₊)` is included elsewhere,
    do not include the multiple in this function
    """
    func(c_e, T) = ones(length(c_e))

    ν_p = func(c_e_p, T_p)
    ν_s = func(c_e_s, T_s)
    ν_n = func(c_e_n, T_n)

    return ν_p, ν_s, ν_n
end

function thermodynamic_factor(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractParam)
    """
    Thermodynamic factor for the activity coefficient. The term `(1-t₊)` is included elsewhere,
    do not include the multiple in this function
    """
    func(c_e, T) = @. (0.601 - 0.24(c_e/1000)^0.5 +0.982*(1-0.0052(T-293))*(c_e/1000)^1.5)

    ν_p = func(c_e_p, T_p)
    ν_s = func(c_e_s, T_s)
    ν_n = func(c_e_n, T_n)

    return ν_p, ν_s, ν_n
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
    if α == 0.5
        j_i = @. 2.0k_i*sqrt(c_e*c_s_star*(c_s_max - c_s_star))*(sinh(0.5*F*η/(R*T)))
    else
        j_i = @. k_i*(c_e)^(1 - α)*c_s_star^α*(c_s_max - c_s_star)^(1 - α)*(exp(α*F*η/(R*T)) - exp(-(α*F*η/(R*T))))
    end

    return j_i
end

function MHC_kfunc(η, λ)
    a = 1.0 + sqrt(λ)

    k = sqrt(π * λ) * (1.0 - erf((λ - sqrt((a) + η ^ 2)) / (2.0 * sqrt(λ)))) / (1.0 + exp(-η))

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

        j_i = @. coeff_rd_ox * (1.0/(1.0 + exp(-η_f))*p.θ[:c_e₀]*c_s_star - 1.0/(1.0 + exp(+η_f))*c_e*c_s_max) * sqrt((1.0 - c_s_star/c_s_max)/p.θ[:c_e₀])
    elseif false

        k_rd = k₀.*MHC_kfunc.(-η_f, λ_MHC)
        k_ox = k₀.*MHC_kfunc.(+η_f, λ_MHC)

        j_i = @. (k_ox*p.θ[:c_e₀]*c_s_star - k_rd*c_e*c_s_max)*sqrt((c_s_max - c_s_star)/(p.θ[:c_e₀] * c_s_max))
    else

        k_rd = k₀.*MHC_kfunc.(-η_f, λ_MHC)
        k_ox = k₀.*MHC_kfunc.(+η_f, λ_MHC)

        act_R = θ_i./(1.0 .- θ_i)

        γ_ts = 1.0./(1.0 .- θ_i) # needed for ecd_extras

        ecd_extras = ĉ_e.^(1.0 .- α) .* act_R.^α ./ (γ_ts.*sqrt.(ĉ_e.*θ_i))


        j_i = ecd_extras.*(k_rd.*ĉ_e .- k_ox.*θ_i)
        j_i .*= (-p.θ[:c_e₀].^(1.0 .- α).*c_s_max) # need this to match dimensional eqn.

        j_i = @. (c_s_max - c_s_star)*((p.θ[:c_e₀]*c_s_star*k_ox - c_e*c_s_max*k_rd)*sqrt((c_s_star*(c_s_max - c_s_star))/(c_s_max*c_e*p.θ[:c_e₀])))
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
