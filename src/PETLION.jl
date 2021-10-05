module PETLION

using StatsBase: mean
using SciMLBase: DAEFunction, DAEProblem, step!, init
using Dierckx: Spline1D
using GeneralizedGenerated: mk_function, RuntimeFn
using LinearAlgebra: diagind, Tridiagonal, norm
using SparseArrays: sparse, findnz, SparseMatrixCSC, spzeros
using SparseDiffTools: matrix_colors, ForwardColorJacCache, forwarddiff_color_jacobian!
using RecursiveArrayTools: VectorOfArray
using Symbolics: @variables, Num, gradient, jacobian_sparsity, expand_derivatives, Differential, get_variables, sparsejacobian, substitute, simplify, build_function, IfElse
using RecipesBase
using SpecialFunctions: erf
using SHA: sha1

import Sundials
import LinearAlgebra
import SuiteSparse

# Must be loaded last
using BSON: @load, @save

export run_model, run_model!
export model_output

export Params
export boundary_stop_conditions, options_model, discretizations_per_section, options_numerical

export D_s_eff_isothermal, D_s_eff
export rxn_rate_isothermal, rxn_rate
export D_eff_linear, D_eff
export K_eff, K_eff_isothermal
export thermodynamic_factor_linear, thermodynamic_factor

export OCV_LCO,  OCV_NMC
export OCV_LiC6, OCV_LiC6_with_NMC

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

end # module