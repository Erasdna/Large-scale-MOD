using LinearAlgebra,ForwardDiff, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools, JLD2
include("../src/LSMOD.jl")
using .LSMOD

"""
    We reproduce Figure 1 in [cite paper]
    Settings:
        - t=2.3 second, 200 timesteps Δt = 10⁻³ and 10⁻⁵
        - At Δt = 10⁻³ we use M=20, m=10
        - At Δt = 10⁻⁵ we use M=35, m=20
        - For each timestep we supply a reference method using the 
          previous step as initial guess for GMRES
        - Other settings:
            - GMRES without restarting
            - Tolerance: ||Ax - b||₂/||b||₂ ≤ 10⁻⁷
            - Incomplete LU preconditioner with no fill in
            - Fourth order discretisation scheme in time
"""

N=1000
M=35
m=20
Nys_k = 14
Nys_p = 6
rows = 2*m
Δt = 1e-5
t₀=0.1
projection_error = true
filename = pwd() * "/Examples/Data/10e_3_full.jld2"

LS_strat(A,rhs,args...) = LSMOD.UniformRowSampledLS(A,rhs,rows,args...)
prob = LSMOD.Example1.make_prob(100)
sol_base,_ = LSMOD.solve(t₀, Δt , N, deepcopy(prob));
RandNYS = LSMOD.Nystrom(prob.internal^2,M,Nys_k,Nys_p);
sol_RNYS = LSMOD.solve(t₀, Δt , N, deepcopy(prob), RandNYS, LS_strat);
pod = LSMOD.POD(prob.internal^2,M,m);
sol_POD = LSMOD.solve(t₀, Δt , N, deepcopy(prob), pod, LS_strat);
RQR=LSMOD.RandomizedQR(prob.internal^2,M,m);
sol_Rand = LSMOD.solve(t₀, Δt , N, deepcopy(prob), RQR, LS_strat);
RSVD=LSMOD.RandomizedSVD(prob.internal^2,M,m);
sol_RandSVD = LSMOD.solve(t₀, Δt, N, deepcopy(prob), RSVD, LS_strat);

save(filename, 
    Dict("base" => sol_base,
         "Nystrom" => sol_RNYS,
         "POD" => sol_POD,
         "RandQR" => sol_Rand,
         "RandSVD" => sol_RandSVD,
         "m" => m,
         "M" => M,
         "dt" => Δt,
         "t0" => t₀,
         "N" => N,
         )
    )
