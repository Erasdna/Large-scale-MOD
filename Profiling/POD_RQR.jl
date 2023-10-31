using LinearAlgebra,ForwardDiff, Revise, Plots, InvertedIndices
include("../src/LSMOD.jl")
using .LSMOD

prob = LSMOD.Example1.prob

N=200
#Δt = 10⁻³
M_3=35
m_3=20
Δt_3 = 1e-3
t₀=0.0

RandNYS = LSMOD.Nystrom(prob.internal^2,M_3,14,7)
#Compilation
@profview LSMOD.solve(t₀, Δt_3 , N, prob, RandNYS)
#Runtime
@profview LSMOD.solve(t₀, Δt_3 , N, prob, RandNYS)