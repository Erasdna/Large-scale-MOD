using LinearAlgebra, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools, JLD2
include("../src/LSMOD.jl")
using .LSMOD

N=1000
M=35
m=20
p = 3

Nys_k = 14
Nys_p = 6

Δt = 1e-3
t₀=0.1

red_N = 50
LS_strategy(A,rhs,args...) = LSMOD.GaussianSketchLS(A,rhs,red_N,args...)

prob = LSMOD.Example1.make_prob(100)
RandNYS = LSMOD.Nystrom(prob.internal^2,M,Nys_k,Nys_p);
sol_RNYS = LSMOD.solve(t₀, Δt , N, deepcopy(prob), RandNYS, LS_strategy);
