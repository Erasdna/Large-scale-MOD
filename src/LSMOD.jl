module LSMOD

using LinearAlgebra, SparseArrays, Base.Iterators, ForwardDiff

include("Problem.jl")
include("Solver.jl")
include("LinearSystem.jl")
include("ReductionStrategies.jl")

end