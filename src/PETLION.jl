module PETLION

using StatsBase: mean
using SciMLBase: DAEFunction, DAEProblem, step!, init
using Dierckx: Spline1D
using GeneralizedGenerated: mk_function, RuntimeFn
using LinearAlgebra: diagind, Tridiagonal, norm
using SparseArrays: sparse, findnz, SparseMatrixCSC, spzeros
using SparseDiffTools: matrix_colors, ForwardColorJacCache, forwarddiff_color_jacobian!
using Parameters: @with_kw
using RecursiveArrayTools: VectorOfArray
using Symbolics: @variables, Num, jacobian_sparsity, expand_derivatives, Differential, get_variables, sparsejacobian, simplify, build_function, IfElse
using RecipesBase
using SpecialFunctions: erf

import Sundials
import LinearAlgebra

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
export OCV_LiC6

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