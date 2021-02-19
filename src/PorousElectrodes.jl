module PorousElectrodes

using LinearAlgebra, Statistics, Sundials, Plots, SparseArrays, ModelingToolkit, SpecialFunctions, ProgressBars, Parameters, SparseDiffTools, RecursiveArrayTools, GeneralizedGenerated
import Dierckx, SymbolicUtils, IfElse
using Base: ImmutableDict
using BSON

export run_model, run_model!, Params, model_output

include("outputs.jl")
include("structures.jl")
include("custom_functions.jl")
include("params.jl")
include("external.jl")
include("model_evaluation.jl")
include("checks.jl")
include("numerical_tools.jl")
include("physics_equations.jl")

nothing

end
