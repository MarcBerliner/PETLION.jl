module PETLION

using LinearAlgebra: AbstractMatrix
using Base: ImmutableDict
using Dierckx: Spline1D
using DifferentialEquations
using GeneralizedGenerated: mk_function, RuntimeFn
using LinearAlgebra
using ModelingToolkit
using SparseArrays
using SparseDiffTools
using Statistics
using Parameters: @with_kw
using ProgressBars: ProgressBar
using RecursiveArrayTools

import IfElse
import Plots
import Sundials

# Must be loaded last
using BSON: @load, @save

export run_model, run_model!
export Params
export model_output

export D_s_eff_isothermal, D_s_eff
export rxn_rate_isothermal, rxn_rate
export D_eff_linear, D_eff
export K_eff, K_eff_isothermal
export thermodynamic_factor_linear, thermodynamic_factor

export OCV_LCO
export OCV_NCA, OCV_NCA_rational_fit_to_error, OCV_NCA_Gaussian, OCV_NCA_Cogswell, OCV_NCA_Tesla
export OCV_LiC6
export OCV_SiC

export rxn_BV, rxn_BV_γMod_01
export rxn_MHC

include("outputs.jl")
include("structures.jl")
include("params.jl")
include("external.jl")
include("set_variables.jl")
include("model_evaluation.jl")
include("checks.jl")
include("generate_functions.jl")
include("physics_equations/residuals.jl")
include("physics_equations/scalar_residual.jl")
include("physics_equations/auxiliary_states_and_coefficients.jl")
include("physics_equations/custom_functions.jl")
include("physics_equations/numerical_tools.jl")

nothing

end
