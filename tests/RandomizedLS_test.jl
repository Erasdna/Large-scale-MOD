using LinearAlgebra, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools, JLD2
include("../src/LSMOD.jl")
using .LSMOD

N=100
M=20
m=10
p = 3

Nys_k = 7
Nys_p = 3

Δt = 1e-5
t₀=0.1

red_N = 50
LS_strategy(A,rhs,args...) = LSMOD.UniformRowSampledLS(A,rhs,red_N,args...)

prob = LSMOD.Example1.make_prob(200)
RandNYS = LSMOD.Nystrom(prob.internal^2,M,Nys_k,Nys_p);
sol_RNYS = LSMOD.solve(t₀, Δt , N, deepcopy(prob), RandNYS);
