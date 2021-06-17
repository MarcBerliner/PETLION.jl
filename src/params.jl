export LCO, LiC6

function NCA_Tesla(θ, funcs)
    ## parameters section
    # everything here can be modified without regenerating the model/jacobian.
    θ[:D_sp] = 1.995860211142953e-14
    θ[:k_p] = 1.0806159336878036e-7
    θ[:λ_MHC_p] = 6.26e-20
    θ[:θ_max_p] = 0.15965575699371948#0.1667
    θ[:θ_min_p] = 0.859486016265424#0.8463
    θ[:l_p] = (143-15)/2*1e-6
    θ[:λ_p] = 2.1
    θ[:ρ_p] = 2500.0
    θ[:Cp_p] = 700.0
    θ[:σ_p] = 100.0
    θ[:ϵ_p] = 0.2297953657980119
    θ[:ϵ_fp] = 0.025
    θ[:brugg_p] = 1.5
    θ[:c_max_p] = 54422.57088984216 # calculated in TeslaOCV.m
    θ[:Rp_p] = 11e-6
    θ[:Ea_D_sp] = 5000.0
    θ[:Ea_k_p] = 5000.0

    funcs.rxn_p = rxn_BV
    funcs.OCV_p = OCV_NCA
end

function LiC6_Tesla(θ, funcs)
    θ[:D_sn] = 1e-14
    θ[:k_n] = 6e-7
    θ[:λ_MHC_n] = 6.26e-20
    θ[:θ_max_n] = 0.9234582496989682#0.9896
    θ[:θ_min_n] = 0.014170948572587282#0.0150
    θ[:l_n] = (174-8)/2*1e-6
    θ[:λ_n] = 1.7
    θ[:ρ_n] = 2500.0
    θ[:Cp_n] = 700.0
    θ[:σ_n] = 100.0
    θ[:ϵ_n] = 0.14730276068895407
    θ[:ϵ_fn] = 0.025
    θ[:brugg_n] = 1.5
    θ[:c_max_n] = 28967.3059327745
    θ[:Rp_n] = 17e-6
    θ[:Ea_D_sn] = 5000.0
    θ[:Ea_k_n] = 5000.0

    θ[:R_film_n] = 0.002 # [Ω⋅m²]

    # Initial SEI resistance value [Ω⋅m²]
    θ[:R_SEI] = 0.01
    #Molar weight                               [kg/mol]
    #ATTENTION: In Development of First Principles Capacity Fade Model
    #for Li-Ion Cells, Ramadass et al. the measurement unit of M_p is wrong as
    #well as the number itself. Please refer to Review of models for predicting
    #the cycling performance of lithium ion batterise, Santhanagopalan et al.
    θ[:M_n] = 73e-3
    # Admittance                                [S/m]
    θ[:k_n_aging] = 3.79e-7
    # Side reaction current density             [A/m²]
    θ[:i_0_jside] = 0.80e-10
    # Open circuit voltage for side reaction    [V]
    θ[:Uref_s] = 0.4
    # Weigthing factor used in the aging dynamics. See the definition of side
    # reaction current density in the ionicFlux.m file.
    θ[:w] = 2.0
    θ[:R_aging] = 1.0 # not sure what to set this to

    funcs.rxn_n = rxn_BV
    funcs.OCV_n = OCV_SiC
end

function θ_System(cathode::typeof(NCA_Tesla), anode::typeof(LiC6_Tesla), θ, funcs;
    # State-of-charge between 0 and 1
    SOC = 1.0,
    # can be any combination of :I, :V, or :P
    methods = (:I, :V),
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
    temperature = false,
    # (:Fickian) Fickian diffusion, (:quadratic) quadratic approx., (:polynomial) polynomial approx.
    solid_diffusion = :Fickian,
    # if solid_diffusion = :Fickian, then this can either be (:finite_difference) or (:spectral)
    Fickian_method = :finite_difference,
    # (false) off, (:SEI) SEI resistance, (:R_aging) constant film
    aging = false,
    # The voltage can be evaluated either at the (:center) center of the finite volume or (:edge) the edge of the finite volume
    edge_values =  :center,
    # (true) symbolic Jacobian, (false) automatic differenation Jacobian
    jacobian = :symbolic,
    ### User-defined functions in `numerics` ###
    # Effective solid diffusion coefficient function
    D_s_eff = D_s_eff_isothermal,
    # Reaction rate function
    rxn_rate = rxn_rate_isothermal,
    # Effective electrolyte diffusion coefficient function
    D_eff = D_eff_EC_EMC_DMC,
    # Effective electrolyte conductivity function
    K_eff = K_eff_EC_EMC_DMC,
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
    θ[:D_e] = 5e-10
    
    θ[:l_s] = 10e-6
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

    θ[:σ_s] = 0.0
    θ[:σ_a] = 3.55e7
    θ[:σ_z] = 5.96e7

    θ[:ϵ_s] = 0.35894212806501313#0.663471282

    θ[:ϵ_fs] = 0.0

    θ[:brugg_s] = 1.5

    θ[:t₊] = 0.455
    θ[:h_cell] = 1.0

    θ[:c_e₀] = 1200.0
    θ[:T₀] = 25 + 273.15
    θ[:T_amb] = 25 + 273.15

    ## Options section
    # everything here can be modified freely

    bounds = boundary_stop_conditions()
    bounds.V_min = 2.7
    bounds.V_max = 4.2
    bounds.SOC_min = 0.0
    bounds.SOC_max = 1.0
    bounds.T_max = 55 + 273.15
    bounds.c_s_n_max = NaN
    bounds.I_max = NaN
    bounds.I_min = NaN


    opts = options_model()
    opts.SOC = SOC # defined above
    opts.outputs = (:t, :V)
    opts.abstol = 1e-8
    opts.reltol = 0.5e-3
    opts.maxiters = 10_000
    opts.check_bounds = true
    opts.reinit = true
    opts.verbose = false
    opts.interp_final = true
    opts.tstops = Float64[]
    opts.tdiscon = Float64[]
    opts.interp_bc = :interpolate
    opts.warm_start = false


    
    #### DO NOT MODIFY BELOW ###
    N = discretizations_per_section(N_p, N_s, N_n, N_a, N_z, N_r_p, N_r_n, -1, -1, -1)
    numerics = options_numerical(cathode, anode, rxn_p, rxn_n, OCV_p, OCV_n, D_s_eff, rxn_rate, D_eff, K_eff, temperature, solid_diffusion, Fickian_method, aging, edge_values, jacobian)
    
    return θ, bounds, opts, N, numerics, methods
end

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

    # Initial SEI resistance value [Ohm m^2]
    θ[:R_SEI] = 0.01
    #Molar weight                               [kg/mol]
    #ATTENTION: In Development of First Principles Capacity Fade Model
    #for Li-Ion Cells, Ramadass et al. the measurement unit of M_p is wrong as
    #well as the number itself. Please refer to Review of models for predicting
    #the cycling performance of lithium ion batterise, Santhanagopalan et al.
    θ[:M_n] = 73e-3
    # Admittance                                [S/m]
    θ[:k_n_aging] = 3.79e-7
    # Side reaction current density             [A/m^2]
    θ[:i_0_jside] = 0.80e-10
    # Open circuit voltage for side reaction    [V]
    θ[:Uref_s] = 0.4
    # Weigthing factor used in the aging dynamics. See the definition of side
    # reaction current density in the ionicFlux.m file.
    θ[:w] = 2.0
    θ[:R_aging] = 1.0 # not sure what to set this to

    funcs.rxn_n = rxn_BV
    funcs.OCV_n = OCV_LiC6
end

function θ_System(cathode::typeof(LCO), anode::typeof(LiC6), θ, funcs;
    # State-of-charge between 0 and 1
    SOC = 1.0,
    # can be any combination of :I, :V, or :P
    methods = (:I, :V, :P),
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
    # The voltage can be evaluated either at the (:center) center of the finite volume or (:edge) the edge of the finite volume
    edge_values =  :center,
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

    θ[:σ_s] = 0.0
    θ[:σ_a] = 3.55e7
    θ[:σ_z] = 5.96e7

    θ[:ϵ_s] = 0.724

    θ[:ϵ_fs] = 0.0

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
    bounds.c_s_n_max = 0.99
    bounds.I_max = NaN
    bounds.I_min = NaN


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
    opts.warm_start = false


    
    #### DO NOT MODIFY BELOW ###
    N = discretizations_per_section(N_p, N_s, N_n, N_a, N_z, N_r_p, N_r_n, -1, -1, -1)
    numerics = options_numerical(cathode, anode, rxn_p, rxn_n, OCV_p, OCV_n, D_s_eff, rxn_rate, D_eff, K_eff, temperature, solid_diffusion, Fickian_method, aging, edge_values, jacobian)
    
    return θ, bounds, opts, N, numerics, methods
end

function Params(;
    cathode::Function = LCO,  # Default chemistry - can be modified
    anode::Function = LiC6, # Default chemistry - can be modified
    kwargs... # keyword arguments for θ_System
    )
    
    θ = Dict{Symbol,Union{Float64,Vector{Float64}}}()
    funcs = _funcs_numerical()

    cathode(θ, funcs)
    anode(θ, funcs)
    θ, bounds, opts, N, numerics, methods = θ_System(cathode, anode, θ, funcs; kwargs...)

    p = initialize_param(θ, bounds, opts, N, numerics, methods)

    return p
end
