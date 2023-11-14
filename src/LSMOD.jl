module LSMOD

using LinearAlgebra, SparseArrays, Base.Iterators, ForwardDiff, TimerOutputs, MKL, LoopVectorization

include("utils.jl")
include("Problem.jl")
include("RandomizedLeastSquares.jl")
include("ReductionStrategies.jl")
include("Solver.jl")
include("LinearSystem.jl")
include("Example_problems.jl")

end