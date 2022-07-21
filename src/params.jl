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
    # Maximum solid particle concentration
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
    # Maximum solid particle concentration
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
    # (:symbolic) symbolic Jacobian, (:AD) automatic differenation Jacobian
    # use symbolic when speed is crucial
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


    opts = options_simulation()
    # Initial state of charge for a new simulation between 0 and 1
    opts.SOC = SOC # defined above
    # Saving sol states is expensive. What states do you want to keep? See the output of solution below for more info. Must be a tuple
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
    # :interpolate or :extrapolate when interpolating the solution
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
    # Maximum solid particle concentration
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
    # Maximum solid particle concentration
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
    # (:symbolic) symbolic Jacobian, (:AD) automatic differenation Jacobian
    # use symbolic when speed is crucial
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


    opts = options_simulation()
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



## NMC_LGM50

export NMC_LGM50
function NMC_LGM50(θ, funcs)
    ## parameters section
    # everything here can be modified without regenerating the sol/jacobian.
    
    # Solid diffusion coefficient [m/s²]
    θ[:D_sp] = 4e-15
    # BV reaction rate constant [m^2.5/(mol^1/2⋅s)]
    θ[:k_p] = 3.5445802224420315e-11
    # MHC reaction, reorganization energy [J] (only needed for MHC reaction)
    θ[:λ_MHC_p] = 0.0
    # Stoichiometry coefficients, θ_min_p > θ_max_p [-]
    θ[:θ_min_p] = 0.8395
    θ[:θ_max_p] = 17038.0/63104.0
    # Thickness of the electrode [m]
    θ[:l_p] = 75.6e-6
    # Conductivity [S/m]
    θ[:σ_p] = 0.18
    # Porosity
    θ[:ϵ_p] = 0.335
    # Filler fraction [note: (active material fraction) = (1 - (porosity) - (filler fraction))]
    θ[:ϵ_fp] = 0.0
    # Bruggeman exponent
    θ[:brugg_p] = 1.5
    # Maximum solid particle concentration
    θ[:c_max_p] = 63104.0
    # Solid particle radius
    θ[:Rp_p] =  5.22e-06


    ## Temperature parameter
    # Thermal conductivity [W/(m⋅K)]
    θ[:λ_p] = 2.1
    # Density [kg/m³]
    θ[:ρ_p] = 3262.0
    # Specific heat capacity [J/(kg⋅K)]
    θ[:Cp_p] = 700.0
    # Activation energy of solid diffusion equation
    θ[:Ea_D_sp] = 0.0
    # Activation energy of reaction rate equation
    θ[:Ea_k_p] = 17800

    ## Stress parameters
    θ[:E_p] = 375e9 # [Pa]
    θ[:ν_p] = 0.3 # [-]
    θ[:Ω_p] = -7.28e-7 # [m³/mol]
    θ[:σ_critical_p] = 375e6 # [Pa]

    ## Custon functions
    # Reaction rate equation
    funcs.rxn_p = rxn_BV
    # Open circuit voltage (OCV or OCP) equation
    function OCV_NMC(θ_p, T=298.15, p=nothing)
        # Define the OCV for the positive electrode
        U_p = @. -0.8090θ_p + 4.4875 - 0.0428 * tanh(18.5138(θ_p - 0.5542)) - 17.7326tanh(15.7890(θ_p - 0.3117)) + 17.5842tanh(15.9308(θ_p - 0.3120))
    
        # Compute the variation of OCV with respect to temperature variations [V/K]
        ∂U∂T_p = zeros(eltype(U_p), length(U_p))
    
        return U_p, ∂U∂T_p
    end
    funcs.OCV_p = OCV_NMC

    return LiC6_LGM50, system_LGM50_NMC_LiC6
end

function LiC6_LGM50(θ, funcs)
    
    # Solid diffusion coefficient [m/s²]
    θ[:D_sn] = 3.3e-14
    # BV reaction rate constant [m^2.5/(mol^1/2⋅s)]
    θ[:k_n] = 6.716046737258585e-12
    # MHC reaction, reorganization energy [J] (only needed for MHC reaction)
    θ[:λ_MHC_n] = 0.0
    # Stoichiometry coefficients, θ_max_n > θ_min_n [-]
    θ[:θ_max_n] = 29866.0/33133
    θ[:θ_min_n] = 0.0481727
    # Thickness of the electrode [m]
    θ[:l_n] = 85.2e-6
    # Conductivity [S/m]
    θ[:σ_n] = 215.0
    # Porosity
    θ[:ϵ_n] = 0.25
    # Filler fraction [note: (active material fraction) = (1 - (porosity) - (filler fraction))]
    θ[:ϵ_fn] = 0.0
    # Bruggeman exponent
    θ[:brugg_n] = 1.5
    # Maximum solid particle concentration
    θ[:c_max_n] = 33133.0
    # Solid particle radius
    θ[:Rp_n] = 5.86e-6

    ## Temperature parameters
    # Thermal conductivity [W/(m⋅K)]
    θ[:λ_n] = 1.7
    # Density [kg/m³]
    θ[:ρ_n] = 1657.0
    # Specific heat capacity [J/(kg⋅K)]
    θ[:Cp_n] = 700.0
    # Activation energy of solid diffusion equation
    θ[:Ea_D_sn] = 3.03e4
    # Activation energy of reaction rate equation
    θ[:Ea_k_n] = 35000.0

    ## Stress parameters
    θ[:c_EC_bulk_n] = 4541.0 # [mol/m³]
    θ[:δ₀] = 5e-9 # [m]
    θ[:V̄_SEI] = 9.585e-5 # [m³/mol]
    θ[:α_SEI] = 0.5 # [-]
    θ[:R_SEI] = 2e5 # [Ω⋅m]
    θ[:E_n] = 15e9 # [Pa]
    θ[:ν_n] = 0.2 # [-]
    θ[:Ω_n] = 3.1e-6 # [m³/mol]
    θ[:σ_critical_n] = 60e6 # [Pa]
    θ[:U_SEI] = 0.4 # [V]
    θ[:k_SEI] = 1e-17 # [m³/mol]
    θ[:D_SEI] = 2e-18 # [Pa]
    

    function OCV_LiC6(θ_n, T=298.15, p=nothing)
        # Define the OCV for the positive electrode
        U_n = @. 1.9793 * exp(-39.3631θ_n) + 0.15561 - 0.0909tanh(29.8538 * (θ_n - 0.1234)) - 0.04478tanh(14.9159 * (θ_n - 0.2769)) - 0.0205tanh(30.4444 * (θ_n - 0.6103)) - 0.09259tanh(17.08 * (θ_n - 1))
        
        # Compute the variation of OCV with respect to temperature variations [V/K]
        ∂U∂T_n = zeros(eltype(U_n), length(U_n))
    
        return U_n, ∂U∂T_n
    end

    ## Custon functions
    # Reaction rate equation
    funcs.rxn_n = rxn_BV
    # Open circuit voltage (OCV or OCP) equation
    funcs.OCV_n = OCV_LiC6
end

D_eff_LGM50(c_e, T, p) = @. p.θ[:D_e] * ((c_e / 1000) ^ 2 - 4.516715942688196 * (c_e / 1000) + 5.5287696156470325)
function D_eff_LGM50(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractModel)
    """
    D_eff evaluates the diffusion coefficients for the electrolyte phase [m^2/s]
    """
    
    D_eff_p = (p.θ[:ϵ_p]^p.θ[:brugg_p])*D_eff_LGM50.(c_e_p, T_p, Ref(p))
    D_eff_s = (p.θ[:ϵ_s]^p.θ[:brugg_s])*D_eff_LGM50.(c_e_s, T_s, Ref(p))
    D_eff_n = (p.θ[:ϵ_n]^p.θ[:brugg_n])*D_eff_LGM50.(c_e_n, T_n, Ref(p))

    return D_eff_p, D_eff_s, D_eff_n
end

K_eff_LGM50(c_e, T, p=nothing) = @. 0.1297 * (c_e / 1000) ^ 3 - 2.51 * (c_e / 1000) ^ 1.5 + 3.329 * (c_e / 1000)
function K_eff_LGM50(c_e_p, c_e_s, c_e_n, T_p, T_s, T_n, p::AbstractModel)
    """
    K_eff evaluates the conductivity coefficients for the electrolyte phase [S/m]
    """

    K_eff_p = (p.θ[:ϵ_p]^p.θ[:brugg_p])*K_eff_LGM50.(c_e_p, T_p)
    K_eff_s = (p.θ[:ϵ_s]^p.θ[:brugg_s])*K_eff_LGM50.(c_e_s, T_s)
    K_eff_n = (p.θ[:ϵ_n]^p.θ[:brugg_n])*K_eff_LGM50.(c_e_n, T_n)

    return K_eff_p, K_eff_s, K_eff_n
end

function system_LGM50_NMC_LiC6(θ, funcs, cathode, anode;
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
    temperature = true,
    # (:Fickian) Fickian diffusion, (:quadratic) quadratic approx., (:polynomial) polynomial approx.
    solid_diffusion = :Fickian,
    # if solid_diffusion = :Fickian, then this can either be (:finite_difference) or (:spectral)
    Fickian_method = :finite_difference,
    # (false) off, (:SEI) SEI resistance
    aging =  :stress,
    # (:symbolic) symbolic Jacobian, (:AD) automatic differenation Jacobian
    # use symbolic when speed is crucial
    jacobian = :symbolic,
    ### User-defined functions in `numerics` ###
    # Effective solid diffusion coefficient function
    D_s_eff = D_s_eff,
    # Reaction rate function
    rxn_rate = rxn_rate,
    # Effective electrolyte diffusion coefficient function
    D_eff = D_eff_LGM50,
    # Effective electrolyte conductivity function
    K_eff = K_eff_LGM50,
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
    θ[:D_e] = 8.794e-11

    # Electrode thicknesses [m/s²]
    θ[:l_s] = 12e-6
    θ[:l_a] = 16e-6
    θ[:l_z] = 12e-6

    
    # Conductivities [S/m]
    θ[:σ_a] = 36.914e6
    θ[:σ_z] = 58.41e6
    
    # Porosity
    θ[:ϵ_s] = 0.47
    
    # Bruggeman exponent
    θ[:brugg_s] = 1.5
    
    # Transference number [-]
    θ[:t₊] = 0.2594
    
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
    θ[:ρ_s] = 397.0
    θ[:ρ_a] = 2700.0
    θ[:ρ_z] = 8960.0

    # Heat capacities [J/(kg⋅K)]
    θ[:Cp_s] = 700.0
    θ[:Cp_a] = 897.0
    θ[:Cp_z] = 385.0

    # Heat transfer coefficient [W/m²⋅K]
    θ[:h_cell] = 1.0

    ## Stress paramters
    θ[:m_LAM] = 2.0 # [-]
    θ[:β_LAM] = 1.9e-6 # [1/s]


    ## Options section
    # everything here can be modified freely

    # `NaN` deactivates the bound
    bounds = boundary_stop_conditions()
    # Maximum permitted voltage [V]
    bounds.V_min = 2.5
    # Minimum permitted voltage [V]
    bounds.V_max = 4.2
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


    opts = options_simulation()
    # Initial state of charge for a new simulation between 0 and 1
    opts.SOC = SOC # defined above
    # Saving sol states is expensive. What states do you want to keep? See the output of solution below for more info. Must be a tuple
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
    # :interpolate or :extrapolate when interpolating the solution
    opts.interp_bc = :interpolate


    
    #### DO NOT MODIFY BELOW ###
    N = discretizations_per_section(N_p, N_s, N_n, N_a, N_z, N_r_p, N_r_n)
    numerics = options_numerical(temperature, solid_diffusion, Fickian_method, aging, cathode, anode, rxn_p, rxn_n, OCV_p, OCV_n, D_s_eff, rxn_rate, D_eff, K_eff, thermodynamic_factor, jacobian)
    
    return θ, bounds, opts, N, numerics
end




function Li_metal(θ, funcs)
    
    # Solid diffusion coefficient [m/s²]
    θ[:D_sn] = 0
    # BV reaction rate constant [m^2.5/(mol^1/2⋅s)]
    θ[:k_n] = 1e-4
    # MHC reaction, reorganization energy [J] (only needed for MHC reaction)
    θ[:λ_MHC_n] = 0.0
    # Stoichiometry coefficients, θ_max_n > θ_min_n [-]
    θ[:θ_max_n] = 0.0
    θ[:θ_min_n] = 1.0
    # Thickness of the electrode [m]
    θ[:l_n] = 25e-6
    # Conductivity [S/m]
    θ[:σ_n] = 1e6
    # Porosity
    θ[:ϵ_n] = 0.25
    # Filler fraction [note: (active material fraction) = (1 - (porosity) - (filler fraction))]
    θ[:ϵ_fn] = 0.0
    # Bruggeman exponent
    θ[:brugg_n] = 1.5
    # Maximum solid particle concentration
    θ[:c_max_n] = 33133.0
    # Solid particle radius
    θ[:Rp_n] = 5.86e-6

    ## Temperature parameters
    # Thermal conductivity [W/(m⋅K)]
    θ[:λ_n] = 1.7
    # Density [kg/m³]
    θ[:ρ_n] = 1657.0
    # Specific heat capacity [J/(kg⋅K)]
    θ[:Cp_n] = 700.0
    # Activation energy of solid diffusion equation
    θ[:Ea_D_sn] = 3.03e4
    # Activation energy of reaction rate equation
    θ[:Ea_k_n] = 35000.0

    ## Stress parameters
    θ[:c_EC_bulk_n] = 4541.0 # [mol/m³]
    θ[:δ₀] = 5e-9 # [m]
    θ[:V̄_SEI] = 9.585e-5 # [m³/mol]
    θ[:α_SEI] = 0.5 # [-]
    θ[:R_SEI] = 2e5 # [Ω⋅m]
    θ[:E_n] = 15e9 # [Pa]
    θ[:ν_n] = 0.2 # [-]
    θ[:Ω_n] = 3.1e-6 # [m³/mol]
    θ[:σ_critical_n] = 60e6 # [Pa]
    θ[:U_SEI] = 0.4 # [V]
    θ[:k_SEI] = 1e-17 # [m³/mol]
    θ[:D_SEI] = 2e-18 # [Pa]
    

    function OCV_Li_metal(θ_n, T=298.15, p=nothing)
        # Define the OCV for the positive electrode
        U_n = @. 1.9793 * exp(-39.3631θ_n) + 0.15561 - 0.0909tanh(29.8538 * (θ_n - 0.1234)) - 0.04478tanh(14.9159 * (θ_n - 0.2769)) - 0.0205tanh(30.4444 * (θ_n - 0.6103)) - 0.09259tanh(17.08 * (θ_n - 1))
        
        # Compute the variation of OCV with respect to temperature variations [V/K]
        ∂U∂T_n = zeros(eltype(U_n), length(U_n))
    
        return U_n, ∂U∂T_n
    end

    ## Custon functions
    # Reaction rate equation
    funcs.rxn_n = rxn_BV
    # Open circuit voltage (OCV or OCP) equation
    funcs.OCV_n = OCV_Li_metal
end