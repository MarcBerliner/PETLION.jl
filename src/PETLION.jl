module PETLION

using Dierckx: Spline1D
using InteractiveUtils: subtypes
using KLU: klu, klu!, KLUFactorization, rcond, klu_factor!
using LinearAlgebra: diagind, Tridiagonal, norm
using OrderedCollections: OrderedDict
using RecipesBase
using RecursiveArrayTools: VectorOfArray
using SciMLBase: DAEFunction, DAEProblem, step!, init
using SHA: sha1
using SparseArrays: sparse, findnz, SparseMatrixCSC, spzeros, spdiagm
using SparseDiffTools: matrix_colors, ForwardColorJacCache, forwarddiff_color_jacobian!
using SpecialFunctions: erf
using Statistics: mean
using Symbolics: @variables, Num, gradient, jacobian_sparsity, expand_derivatives, Differential, get_variables, sparsejacobian, substitute, simplify, build_function, IfElse, SerialForm

import LinearAlgebra
import PkgVersion
import ProgressMeter
import SuiteSparse
import Sundials
import Symbolics

# Must be loaded last
using BSON: @load, @save

export simulate, simulate!
export solution
export final_exit_reason, exit_reasons

export petlion
export boundary_stop_conditions, options_simulation, discretizations_per_section, options_numerical

export D_s_eff_isothermal, D_s_eff
export rxn_rate_isothermal, rxn_rate
export D_eff_linear, D_eff
export K_eff, K_eff_isothermal
export thermodynamic_factor_linear, thermodynamic_factor

export OCV_LCO,  OCV_NMC
export OCV_LiC6, OCV_LiC6_with_NMC

export rxn_BV, rxn_BV_γMod_01
export rxn_MHC


include("states_definition.jl")
include("outputs.jl")
include("structures.jl")
include("params.jl")
include("external.jl")
include("model_evaluation.jl")
include("generate_functions.jl")
include("physics_equations/residuals.jl")
include("physics_equations/scalar_residual.jl")
include("physics_equations/auxiliary_states_and_coefficients.jl")
include("physics_equations/custom_functions.jl")
include("physics_equations/numerical_tools.jl")
include("physics_equations/input_methods.jl")
include("checks.jl")
include("save_outputs.jl")

## Backwards compatability
Params(x...;kw...) = petlion(x...;kw...)
run_model(x...;kw...) = simulate(x...;kw...)
run_model!(x...;kw...) = simulate!(x...;kw...)
options_model(x...;kw...) = options_simulation(x...;kw...)
model_output(x...;kw...) = solution(x...;kw...)

## Deprecation warning
@deprecate Params(cathode) petlion(cathode)
@deprecate run_model(p,x...;kw...) simulate(p,x...;kw...)
@deprecate run_model!(sol,p,x...;kw...) simulate!(sol,p,x...;kw...)
@deprecate options_model() options_simulation()
@deprecate model_output() solution()


export Params, run_model, run_model!, options_model, model_output

end # module
