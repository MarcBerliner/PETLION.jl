export LCO, NMC

## LCO

function LCO(θ, funcs)
    ## parameters section
    # everything here can be modified without regenerating the sol/jacobian.
    
    # Solid diffusion coefficient [m/s²]
    θ[:D_sp] = 1e-14
    # Electrolyte diffusion coefficient [m/s²]
    θ[:D_p] = 7.5e-10
    # BV reaction rate constant [m^2.5/(mol^1/2⋅s)]
    θ[:k_p] = 2.334e-11
    # MHC reaction, reorganization energy [J] (only needed for MHC reaction)
    θ[:λ_MHC_p] = 6.26e-20
    # Stoichiometry coefficients, θ_min_p > θ_max_p [-]
    θ[:θ_min_p] = 0.99174
    θ[:θ_max_p] = 0.49550
    # Thickness of the electrode [m]
    θ[:l_p] = 80e-6
    # Conductivity [S/m]
    θ[:σ_p] = 100.0
    # Porosity
    θ[:ϵ_p] = 0.385
    # Filler fraction [note: (active material fraction) = (1 - (porosity) - (filler fraction))]
    θ[:ϵ_fp] = 0.025
    # Bruggeman exponent
    θ[:brugg_p] = 4.0
    # Maximum solid interpolate_electrolyte_concentration
    θ[:c_max_p] = 51554.0
    # Solid particle radius
    θ[:Rp_p] = 2e-6


    ## Temperature parameter
    # Thermal conductivity [W/(m⋅K)]
    θ[:λ_p] = 2.1
    # Density [kg/m³]
    θ[:ρ_p] = 2500.0
    # Specific heat capacity [J/(kg⋅K)]
    θ[:Cp_p] = 700.0
    # Activation energy of solid diffusion equation
    θ[:Ea_D_sp] = 5000.0
    # Activation energy of reaction rate equation
    θ[:Ea_k_p] = 5000.0


    ## Custon functions
    # Reaction rate equation
    funcs.rxn_p = rxn_BV
    # Open circuit voltage (OCV or OCP) equation
    funcs.OCV_p = OCV_LCO

    return LiC6, system_LCO_LiC6
end

function LiC6(θ, funcs)
    
    # Solid diffusion coefficient [m/s²]
    θ[:D_sn] = 3.9e-14
    # Electrolyte diffusion coefficient [m/s²]
    θ[:D_n] = 7.5e-10
    # BV reaction rate constant [m^2.5/(mol^1/2⋅s)]
    θ[:k_n] = 5.0310e-11
    # MHC reaction, reorganization energy [J] (only needed for MHC reaction)
    θ[:λ_MHC_n] = 6.26e-20
    # Stoichiometry coefficients, θ_max_n > θ_min_n [-]
    θ[:θ_max_n] = 0.85510
    θ[:θ_min_n] = 0.01429
    # Thickness of the electrode [m]
    θ[:l_n] = 88e-6
    # Conductivity [S/m]
    θ[:σ_n] = 100.0
    # Porosity
    θ[:ϵ_n] = 0.485
    # Filler fraction [note: (active material fraction) = (1 - (porosity) - (filler fraction))]
    θ[:ϵ_fn] = 0.0326
    # Bruggeman exponent
    θ[:brugg_n] = 4.0
    # Maximum solid interpolate_electrolyte_concentration
    θ[:c_max_n] = 30555.0
    # Solid particle radius
    θ[:Rp_n] = 2e-6

    ## Temperature parameters
    # Thermal conductivity [W/(m⋅K)]
    θ[:λ_n] = 1.7
    # Density [kg/m³]
    θ[:ρ_n] = 2500.0
    # Specific heat capacity [J/(kg⋅K)]
    θ[:Cp_n] = 700.0
    # Activation energy of solid diffusion equation
    θ[:Ea_D_sn] = 5000.0
    # Activation energy of reaction rate equation
    θ[:Ea_k_n] = 5000.0

    ## Aging parameters
    # Initial SEI resistance value [Ω⋅m²]
    θ[:R_SEI] = 0.01
    # Molar weight [kg/mol]
    θ[:M_n] = 7.3e-4
    # Admittance [S/m]
    θ[:k_n_aging] = 1.0
    # Side reaction current density [A/m²]
    θ[:i_0_jside] = 1.5e-6
    # Open circuit voltage for side reaction [V]
    θ[:Uref_s] = 0.4
    # Weigthing factor used in the aging dynamics
    θ[:w] = 2.0
    
    ## Custon functions
    # Reaction rate equation
    funcs.rxn_n = rxn_BV
    # Open circuit voltage (OCV or OCP) equation
    funcs.OCV_n = OCV_LiC6
end

function system_LCO_LiC6(θ, funcs, cathode, anode;
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
    # (false) off, (:SEI) SEI resistance
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
    # By default, this
    ## Custon functions
    # Reaction rate equation will use the reaction defined by the cathode
    rxn_p = funcs.rxn_p,
    # Open circuit voltage (OCV or OCP) equation
    # By default, this will use the OCV defined by the cathode
    OCV_p = funcs.OCV_p,
    # By default, this
    ## Custon functions
    # Reaction rate equation will use the reaction defined by the anode
    rxn_n = funcs.rxn_n,
    # Open circuit voltage (OCV or OCP) equation
    # By default, this will use the OCV defined by the anode
    OCV_n = funcs.OCV_n,
    )


    ## Physical parameters for the system
    
    # Electrolyte diffusion coefficient [m/s²]
    θ[:D_s] = 7.5e-10

    # Electrode thicknesses [m/s²]
    θ[:l_s] = 25e-6
    θ[:l_a] = 10e-6
    θ[:l_z] = 10e-6

    
    # Conductivities [S/m]
    θ[:σ_a] = 3.55e7
    θ[:σ_z] = 5.96e7
    
    # Porosity
    θ[:ϵ_s] = 0.724
    
    # Bruggeman exponent
    θ[:brugg_s] = 4.0
    
    # Transference number [-]
    θ[:t₊] = 0.364
    
    # Initial electrolyte concentration [mol/m³]
    θ[:c_e₀] = 1000.0
    # Initial temperature [K]
    θ[:T₀] = 25 + 273.15
    # Ambient temperature [K]
    θ[:T_amb] = 25 + 273.15


    ## Temperature
    # Thermal conductivities [W/(m⋅K)]
    θ[:λ_s] = 0.16
    θ[:λ_a] = 237.0
    θ[:λ_z] = 401.0

    # Densities [kg/m³]
    θ[:ρ_s] = 1100.0
    θ[:ρ_a] = 2700.0
    θ[:ρ_z] = 8940.0

    # Heat capacities [J/(kg⋅K)]
    θ[:Cp_s] = 700.0
    θ[:Cp_a] = 897.0
    θ[:Cp_z] = 385.0

    # Heat transfer coefficient [W/m²⋅K]
    θ[:h_cell] = 1.0


    ## Options section
    # everything here can be modified freely

    # `NaN` deactivates the bound
    bounds = boundary_stop_conditions()
    # Maximum permitted voltage [V]
    bounds.V_min = 2.5
    # Minimum permitted voltage [V]
    bounds.V_max = 4.3
    # Maximum permitted SOC [-]
    bounds.SOC_min = 0.0
    # Minimum permitted SOC [-]
    bounds.SOC_max = 1.0
    # Maximum permitted temperature [K]
    bounds.T_max = 55 + 273.15
    # Maximum permitted solid surface concentration in the anode [mol/m³]
    bounds.c_s_n_max = NaN
    # Maximum permitted current [C-rate]
    bounds.I_max = NaN
    # Minimum permitted current [C-rate]
    bounds.I_min = NaN
    # Minimum permitted plating overpotential at the separator-anode interface [V]
    bounds.η_plating_min = NaN
    # Minimum permitted electrolyte concentration [mol/m³]
    bounds.c_e_min = NaN


    opts = options_model()
    # Initial state of charge for a new simulation between 0 and 1
    opts.SOC = SOC # defined above
    # Saving sol states is expensive. What states do you want to keep? See the output of sol below for more info. Must be a tuple
    opts.outputs = (:t, :V)
    # Absolute tolerance of DAE solver
    opts.abstol = 1e-6
    # Relative tolerance of DAE solver
    opts.reltol = 1e-3
    # Maximum iterations for the DAE solver
    opts.maxiters = 10_000
    # Flag to check the bounds during simulation (SOC max/min, V max/min, etc.)
    opts.check_bounds = true
    # Get a new initial guess for DAE initialization
    opts.reinit = true
    # Show some outputs during simulation runtime
    opts.verbose = false
    # Interpolate the final results to match the exact simulation end point
    opts.interp_final = true
    # Times when the DAE solver explicitly stops
    opts.tstops = Float64[]
    # For input functions, times when there is a known discontinuity. Unknown discontinuities are handled automatically but less efficiently
    opts.tdiscon = Float64[]
    # :interpolate or :extrapolate when interpolating the sol
    opts.interp_bc = :interpolate


    
    #### DO NOT MODIFY BELOW ###
    N = discretizations_per_section(N_p, N_s, N_n, N_a, N_z, N_r_p, N_r_n)
    numerics = options_numerical(temperature, solid_diffusion, Fickian_method, aging, cathode, anode, rxn_p, rxn_n, OCV_p, OCV_n, D_s_eff, rxn_rate, D_eff, K_eff, thermodynamic_factor, jacobian)
    
    return θ, bounds, opts, N, numerics
end



## NMC

function NMC(θ, funcs)
    ## parameters section
    # everything here can be modified without regenerating the sol/jacobian.
    
    # Solid diffusion coefficient [m/s²]
    θ[:D_sp] = 2e-14
    # BV reaction rate constant [m^2.5/(mol^1/2⋅s)]
    θ[:k_p] = 6.3066e-10
    # Stoichiometry coefficients, θ_min_p > θ_max_p [-]
    θ[:θ_min_p] = 0.955473
    θ[:θ_max_p] = 0.359749
    # Thickness of the electrode [m]
    θ[:l_p] = 41.6e-6
    # Conductivity [S/m]
    θ[:σ_p] = 100
    # Porosity
    θ[:ϵ_p] = 0.3
    # Filler fraction [note: (active material fraction) = (1 - (porosity) - (filler fraction))]
    θ[:ϵ_fp] = 0.12
    # Bruggeman exponent
    θ[:brugg_p] = 1.5
    # Maximum solid interpolate_electrolyte_concentration
    θ[:c_max_p] = 51830.0
    # Solid particle radius
    θ[:Rp_p] = 7.5e-6
    # Activation energy of solid diffusion equation
    θ[:Ea_D_sp] = 2.5e4
    # Activation energy of reaction rate equation
    θ[:Ea_k_p] = 3e4

    ## Custon functions
    # Reaction rate equation
    funcs.rxn_p = rxn_BV
    # Open circuit voltage (OCV or OCP) equation
    funcs.OCV_p = OCV_NMC

    return LiC6_NMC, system_NMC_LiC6
end

function LiC6_NMC(θ, funcs)
    
    # Solid diffusion coefficient [m/s²]
    θ[:D_sn] = 1.5e-14
    # BV reaction rate constant [m^2.5/(mol^1/2⋅s)]
    θ[:k_n] = 6.3466e-10
    # Stoichiometry coefficients, θ_max_n > θ_min_n [-]
    θ[:θ_max_n] = 0.790813
    θ[:θ_min_n] = 0.001
    # Thickness of the electrode [m]
    θ[:l_n] = 48e-6
    # Conductivity [S/m]
    θ[:σ_n] = 100
    # Porosity
    θ[:ϵ_n] = 0.3
    # Filler fraction [note: (active material fraction) = (1 - (porosity) - (filler fraction))]
    θ[:ϵ_fn] = 0.038
    # Bruggeman exponent
    θ[:brugg_n] = 1.5
    # Maximum solid interpolate_electrolyte_concentration
    θ[:c_max_n] = 31080.0
    # Solid particle radius
    θ[:Rp_n] = 10e-6
    # Activation energy of solid diffusion equation
    θ[:Ea_D_sn] = 4e4
    # Activation energy of reaction rate equation
    θ[:Ea_k_n] = 3e4

    ## Custon functions
    # Reaction rate equation
    funcs.rxn_n = rxn_BV
    # Open circuit voltage (OCV or OCP) equation
    funcs.OCV_n = OCV_LiC6_with_NMC
end

function system_NMC_LiC6(θ, funcs, cathode, anode;
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
    # (false) off, (:SEI) SEI resistance
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
    thermodynamic_factor = thermodynamic_factor_linear,
    # By default, this
    ## Custon functions
    # Reaction rate equation will use the reaction defined by the cathode
    rxn_p = funcs.rxn_p,
    # Open circuit voltage (OCV or OCP) equation
    # By default, this will use the OCV defined by the cathode
    OCV_p = funcs.OCV_p,
    # By default, this
    ## Custon functions
    # Reaction rate equation will use the reaction defined by the anode
    rxn_n = funcs.rxn_n,
    # Open circuit voltage (OCV or OCP) equation
    # By default, this will use the OCV defined by the anode
    OCV_n = funcs.OCV_n,
    )


    
    # Electrode thicknesses [m/s²]## Physical parameters for the system
    θ[:l_s] = 25e-6

    # Porosity
    θ[:ϵ_s] = 0.4

    # Thermal conductivities [W/(m⋅K)]
    θ[:brugg_s] = 1.5

    # Transference number [-]
    θ[:t₊] = 0.38

    # Initial electrolyte concentration [mol/m³]
    θ[:c_e₀] = 1200
    # Initial temperature [K]
    θ[:T₀] = 25 + 273.15
    # Ambient temperature [K]
    θ[:T_amb] = 25 + 273.15

    ## Options section
    # everything here can be modified freely

    # `NaN` deactivates the bound
    bounds = boundary_stop_conditions()
    # Maximum permitted voltage [V]
    bounds.V_min = 2.8
    # Minimum permitted voltage [V]
    bounds.V_max = 4.2
    # Maximum permitted SOC [-]
    bounds.SOC_min = 0.0
    # Minimum permitted SOC [-]
    bounds.SOC_max = 1.0
    # Maximum permitted temperature [K]
    bounds.T_max = NaN
    # Maximum permitted solid surface concentration in the anode [mol/m³]
    bounds.c_s_n_max = NaN
    # Maximum permitted current [C-rate]
    bounds.I_max = NaN
    # Minimum permitted current [C-rate]
    bounds.I_min = NaN
    # Minimum permitted plating overpotential at the separator-anode interface [V]
    bounds.η_plating_min = NaN
    # Minimum permitted electrolyte concentration [mol/m³]
    bounds.c_e_min = NaN


    opts = options_model()
    # Initial state of charge for a new simulation between 0 and 1
    opts.SOC = SOC # defined above
    # Saving sol states is expensive. What states do you want to keep? See the output of sol below for more info. Must be a tuple
    opts.outputs = (:t, :V)
    # Absolute tolerance of DAE solver
    opts.abstol = 1e-6
    # Relative tolerance of DAE solver
    opts.reltol = 1e-3
    # Maximum iterations for the DAE solver
    opts.maxiters = 10_000
    # Flag to check the bounds during simulation (SOC max/min, V max/min, etc.)
    opts.check_bounds = true
    # Get a new initial guess for DAE initialization
    opts.reinit = true
    # Show some outputs during simulation runtime
    opts.verbose = false
    # Interpolate the final results to match the exact simulation end point
    opts.interp_final = true
    # Times when the DAE solver explicitly stops
    opts.tstops = Float64[]
    # For input functions, times when there is a known discontinuity. Unknown discontinuities are handled automatically but less efficiently
    opts.tdiscon = Float64[]
    # :interpolate or :extrapolate when interpolating the sol
    opts.interp_bc = :interpolate


    
    #### DO NOT MODIFY BELOW ###
    N = discretizations_per_section(N_p, N_s, N_n, N_a, N_z, N_r_p, N_r_n)
    numerics = options_numerical(temperature, solid_diffusion, Fickian_method, aging, cathode, anode, rxn_p, rxn_n, OCV_p, OCV_n, D_s_eff, rxn_rate, D_eff, K_eff, thermodynamic_factor, jacobian)
    
    return θ, bounds, opts, N, numerics
end

