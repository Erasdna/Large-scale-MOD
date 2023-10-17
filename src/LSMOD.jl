module LSMOD

using LinearAlgebra, SparseArrays, Base.Iterators, ForwardDiff, TimerOutputs, MKL

include("utils.jl")
include("Problem.jl")
include("Solver.jl")
include("LinearSystem.jl")
include("ReductionStrategies.jl")

end