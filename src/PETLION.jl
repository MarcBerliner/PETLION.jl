module PETLION

cd(@__DIR__ )

using Base: ImmutableDict
using Dierckx: Spline1D
using DifferentialEquations
using GeneralizedGenerated: mk_function, RuntimeFn
using IfElse: ifelse
using LinearAlgebra
using ModelingToolkit
using SparseArrays
using SparseDiffTools
using SpecialFunctions
using Statistics
using Parameters: @with_kw
using ProgressBars: ProgressBar
using RecursiveArrayTools

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
export OCV_LCO, OCV_NCA, OCV_NCA_rational_fit_to_error, OCV_NCA_Gaussian, OCV_NCA_Cogswell, OCV_NCA_Tesla, OCV_LiC6, OCV_SiC
export rxn_BV, rxn_MHC, rxn_BV_γMod_01

include("outputs.jl")
include("structures.jl")
include("custom_functions.jl")
include("params.jl")
include("external.jl")
include("set_variables.jl")
include("model_evaluation.jl")
include("checks.jl")
include("numerical_tools.jl")
include("physics_equations.jl")
include("generate_functions.jl")

nothing

end
