export LCO, NMC

function LCO(θ, funcs)
    ## parameters section
    # everything here can be modified without regenerating the model/jacobian.
    θ[:D_sp] = 1e-14
    θ[:D_p] = 7.5e-10
    θ[:k_p] = 2.334e-11
    θ[:λ_MHC_p] = 6.26e-20
    θ[:θ_max_p] = 0.49550
    θ[:θ_min_p] = 0.99174
    θ[:l_p] = 80e-6
    θ[:λ_p] = 2.1
    θ[:ρ_p] = 2500.0
    θ[:Cp_p] = 700.0
    θ[:σ_p] = 100.0
    θ[:ϵ_p] = 0.385
    θ[:ϵ_fp] = 0.025
    θ[:brugg_p] = 4.0
    θ[:c_max_p] = 51554.0
    θ[:Rp_p] = 2e-6
    θ[:Ea_D_sp] = 5000.0
    θ[:Ea_k_p] = 5000.0

    funcs.rxn_p = rxn_BV
    funcs.OCV_p = OCV_LCO
end

function LiC6(θ, funcs)
    θ[:D_sn] = 3.9e-14
    θ[:D_n] = 7.5e-10
    θ[:k_n] = 5.0310e-11
    θ[:λ_MHC_n] = 6.26e-20
    θ[:θ_max_n] = 0.85510
    θ[:θ_min_n] = 0.01429
    θ[:l_n] = 88e-6
    θ[:λ_n] = 1.7
    θ[:ρ_n] = 2500.0
    θ[:Cp_n] = 700.0
    θ[:σ_n] = 100.0
    θ[:ϵ_n] = 0.485
    θ[:ϵ_fn] = 0.0326
    θ[:brugg_n] = 4.0
    θ[:c_max_n] = 30555.0
    θ[:Rp_n] = 2e-6
    θ[:Ea_D_sn] = 5000.0
    θ[:Ea_k_n] = 5000.0

    # Initial SEI resistance value [Ω⋅m²]
    θ[:R_SEI] = 0.01
    #Molar weight                               [kg/mol]
    #ATTENTION: In Development of First Principles Capacity Fade Model
    #for Li-Ion Cells, Ramadass et al. the measurement unit of M_p is wrong as
    #well as the number itself. Please refer to Review of models for predicting
    #the cycling performance of lithium ion batterise, Santhanagopalan et al.
    θ[:M_n] = 7.3e-4
    # Admittance                                [S/m]
    θ[:k_n_aging] = 1.0
    # Side reaction current density             [A/m²]
    θ[:i_0_jside] = 1.5e-6
    # Open circuit voltage for side reaction    [V]
    θ[:Uref_s] = 0.4
    # Weigthing factor used in the aging dynamics. See the definition of side
    # reaction current density in the ionicFlux.m file.
    θ[:w] = 2.0
    θ[:R_aging] = 0.01 # not sure what to set this to

    funcs.rxn_n = rxn_BV
    funcs.OCV_n = OCV_LiC6
end

function NMC(θ, funcs)
    ## parameters section
    # everything here can be modified without regenerating the model/jacobian.
    θ[:D_sp] = 2e-14
    θ[:k_p] = 6.3066e-10
    θ[:θ_max_p] = 0.359749
    θ[:θ_min_p] = 0.955473
    θ[:l_p] = 41.6e-6
    θ[:σ_p] = 100
    θ[:ϵ_p] = 0.3
    θ[:ϵ_fp] = 0.12
    θ[:brugg_p] = 1.5
    θ[:c_max_p] = 51830.0
    θ[:Rp_p] = 7.5e-6
    θ[:Ea_D_sp] = 2.5e4
    θ[:Ea_k_p] = 3e4

    funcs.rxn_p = rxn_BV
    funcs.OCV_p = OCV_NMC
end

function LiC6_NMC(θ, funcs)
    θ[:D_sn] = 1.5e-14
    θ[:k_n] = 6.3466e-10
    θ[:θ_max_n] = 0.790813
    θ[:θ_min_n] = 0.001
    θ[:l_n] = 48e-6
    θ[:σ_n] = 100
    θ[:ϵ_n] = 0.3
    θ[:ϵ_fn] = 0.038
    θ[:brugg_n] = 1.5
    θ[:c_max_n] = 31080.0
    θ[:Rp_n] = 10e-6
    θ[:Ea_D_sn] = 4e4
    θ[:Ea_k_n] = 3e4

    funcs.rxn_n = rxn_BV
    funcs.OCV_n = OCV_LiC6_with_NMC
end

function θ_System(cathode::typeof(NMC), anode::typeof(LiC6_NMC), θ, funcs;
    # State-of-charge between 0 and 1
    SOC = 1.0,
    ### Cell discretizations, `N` ###
    # Volume discretizations per cathode
    N_p = 10,
    # Volume discretizations per separator
    N_s = 10,
    # Volume discretizations per anode
    N_n = 10,
    # Volume discretizations per positive current collector (temperature only)
    N_a = 10,
    # Volume discretizations per negative current collector (temperature only)
    N_z = 10,
    # Volume discretizations per cathode particle (Fickian diffusion only)
    N_r_p = 10,
    # Volume discretizations per anode particle (Fickian diffusion only)
    N_r_n = 10,
    ### Numerical options, `numerics` ###
    # 1D temperature, true or false
    temperature =  false,
    # (:Fickian) Fickian diffusion, (:quadratic) quadratic approx., (:polynomial) polynomial approx.
    solid_diffusion = :Fickian,
    # if solid_diffusion = :Fickian, then this can either be (:finite_difference) or (:spectral)
    Fickian_method = :finite_difference,
    # (false) off, (:SEI) SEI resistance, (:R_aging) constant film
    aging =  false,
    # (true) symbolic Jacobian, (false) automatic differenation Jacobian
    jacobian = :symbolic,
    ### User-defined functions in `numerics` ###
    # Effective solid diffusion coefficient function
    D_s_eff = D_s_eff,
    # Reaction rate function
    rxn_rate = rxn_rate,
    # Effective electrolyte diffusion coefficient function
    D_eff = D_eff,
    # Effective electrolyte conductivity function
    K_eff = K_eff,
    # Thermodynamic factor, ∂ln(f)/∂ln(c_e)
    thermodynamic_factor = thermodynamic_factor,
    # By default, this will use the reaction defined by the cathode
    rxn_p = funcs.rxn_p,
    # By default, this will use the OCV defined by the cathode
    OCV_p = funcs.OCV_p,
    # By default, this will use the reaction defined by the anode
    rxn_n = funcs.rxn_n,
    # By default, this will use the OCV defined by the anode
    OCV_n = funcs.OCV_n,
    )


    ## Physical parameters for the system
    θ[:l_s] = 25e-6

    θ[:ϵ_s] = 0.4

    θ[:brugg_s] = 1.5

    θ[:t₊] = 0.38

    θ[:c_e₀] = 1200
    θ[:T₀] = 25 + 273.15
    θ[:T_amb] = 25 + 273.15

    ## Options section
    # everything here can be modified freely

    bounds = boundary_stop_conditions()
    bounds.V_min = 2.8
    bounds.V_max = 4.2
    bounds.SOC_min = 0.0
    bounds.SOC_max = 1.0
    bounds.T_max = NaN
    bounds.c_s_n_max = NaN
    bounds.I_max = NaN
    bounds.I_min = NaN
    bounds.η_plating_min = NaN
    bounds.c_e_min = NaN


    opts = options_model()
    opts.SOC = SOC # defined above
    opts.outputs = (:t, :V)
    opts.abstol = 1e-6
    opts.reltol = 1e-3
    opts.maxiters = 10_000
    opts.check_bounds = true
    opts.reinit = true
    opts.verbose = false
    opts.interp_final = true
    opts.tstops = Float64[]
    opts.tdiscon = Float64[]
    opts.interp_bc = :interpolate
    opts.save_start = false


    
    #### DO NOT MODIFY BELOW ###
    N = discretizations_per_section(N_p, N_s, N_n, N_a, N_z, N_r_p, N_r_n, -1, -1, -1)
    numerics = options_numerical(temperature, solid_diffusion, Fickian_method, aging, cathode, anode, rxn_p, rxn_n, OCV_p, OCV_n, D_s_eff, rxn_rate, D_eff, K_eff, thermodynamic_factor, jacobian)
    
    return θ, bounds, opts, N, numerics
end

function θ_System(cathode::typeof(LCO), anode::typeof(LiC6), θ, funcs;
    # State-of-charge between 0 and 1
    SOC = 1.0,
    ### Cell discretizations, `N` ###
    # Volume discretizations per cathode
    N_p = 10,
    # Volume discretizations per separator
    N_s = 10,
    # Volume discretizations per anode
    N_n = 10,
    # Volume discretizations per positive current collector (temperature only)
    N_a = 10,
    # Volume discretizations per negative current collector (temperature only)
    N_z = 10,
    # Volume discretizations per cathode particle (Fickian diffusion only)
    N_r_p = 10,
    # Volume discretizations per anode particle (Fickian diffusion only)
    N_r_n = 10,
    ### Numerical options, `numerics` ###
    # 1D temperature, true or false
    temperature =  false,
    # (:Fickian) Fickian diffusion, (:quadratic) quadratic approx., (:polynomial) polynomial approx.
    solid_diffusion = :Fickian,
    # if solid_diffusion = :Fickian, then this can either be (:finite_difference) or (:spectral)
    Fickian_method = :finite_difference,
    # (false) off, (:SEI) SEI resistance, (:R_aging) constant film
    aging =  false,
    # (true) symbolic Jacobian, (false) automatic differenation Jacobian
    jacobian = :symbolic,
    ### User-defined functions in `numerics` ###
    # Effective solid diffusion coefficient function
    D_s_eff = D_s_eff,
    # Reaction rate function
    rxn_rate = rxn_rate,
    # Effective electrolyte diffusion coefficient function
    D_eff = D_eff_linear,
    # Effective electrolyte conductivity function
    K_eff = K_eff,
    # Thermodynamic factor, ∂ln(f)/∂ln(c_e)
    thermodynamic_factor = thermodynamic_factor_linear,
    # By default, this will use the reaction defined by the cathode
    rxn_p = funcs.rxn_p,
    # By default, this will use the OCV defined by the cathode
    OCV_p = funcs.OCV_p,
    # By default, this will use the reaction defined by the anode
    rxn_n = funcs.rxn_n,
    # By default, this will use the OCV defined by the anode
    OCV_n = funcs.OCV_n,
    )


    ## Physical parameters for the system
    θ[:D_s] = 7.5e-10
    
    θ[:l_s] = 25e-6
    θ[:l_a] = 10e-6
    θ[:l_z] = 10e-6

    θ[:λ_s] = 0.16
    θ[:λ_a] = 237.0
    θ[:λ_z] = 401.0

    θ[:ρ_s] = 1100.0
    θ[:ρ_a] = 2700.0
    θ[:ρ_z] = 8940.0

    θ[:Cp_s] = 700.0
    θ[:Cp_a] = 897.0
    θ[:Cp_z] = 385.0

    θ[:σ_a] = 3.55e7
    θ[:σ_z] = 5.96e7

    θ[:ϵ_s] = 0.724

    θ[:brugg_s] = 4.0

    θ[:t₊] = 0.364
    θ[:h_cell] = 1.0

    θ[:c_e₀] = 1000.0
    θ[:T₀] = 25 + 273.15
    θ[:T_amb] = 25 + 273.15

    ## Options section
    # everything here can be modified freely

    bounds = boundary_stop_conditions()
    bounds.V_min = 2.5
    bounds.V_max = 4.3
    bounds.SOC_min = 0.0
    bounds.SOC_max = 1.0
    bounds.T_max = 55 + 273.15
    bounds.c_s_n_max = NaN
    bounds.I_max = NaN
    bounds.I_min = NaN
    bounds.η_plating_min = NaN
    bounds.c_e_min = NaN


    opts = options_model()
    opts.SOC = SOC # defined above
    opts.outputs = (:t, :V)
    opts.abstol = 1e-6
    opts.reltol = 1e-3
    opts.maxiters = 10_000
    opts.check_bounds = true
    opts.reinit = true
    opts.verbose = false
    opts.interp_final = true
    opts.tstops = Float64[]
    opts.tdiscon = Float64[]
    opts.interp_bc = :interpolate
    opts.save_start = false


    
    #### DO NOT MODIFY BELOW ###
    N = discretizations_per_section(N_p, N_s, N_n, N_a, N_z, N_r_p, N_r_n, -1, -1, -1)
    numerics = options_numerical(temperature, solid_diffusion, Fickian_method, aging, cathode, anode, rxn_p, rxn_n, OCV_p, OCV_n, D_s_eff, rxn_rate, D_eff, K_eff, thermodynamic_factor, jacobian)
    
    return θ, bounds, opts, N, numerics
end

function Params(chemistry::typeof(LCO);kwargs...)
    cathode = LCO
    anode   = LiC6
    return Params(;cathode=cathode,anode=anode,kwargs...)
end
function Params(chemistry::typeof(NMC);kwargs...)
    cathode = NMC
    anode   = LiC6_NMC
    return Params(;cathode=cathode,anode=anode,kwargs...)
end

function Params(;cathode=cathode,anode=anode, # Input chemistry - can be modified
    kwargs... # keyword arguments for θ_System
    )
    
    θ = Dict{Symbol,Float64}()
    funcs = _funcs_numerical()

    cathode(θ, funcs)
    anode(θ, funcs)
    θ, bounds, opts, N, numerics = θ_System(cathode, anode, θ, funcs; kwargs...)

    p = initialize_param(θ, bounds, opts, N, numerics)

    return p
end
