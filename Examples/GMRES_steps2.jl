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
M=20
m=10

Nys_k = 7
Nys_p = 3
Δt = 1e-5
t₀=0.0
projection_error = true
filename = pwd() * "/Examples/Data/10e_5_all.jld2"

prob = LSMOD.Example1.make_prob(100)

sol_base,_ = LSMOD.solve(t₀, Δt , N, prob)
RandNYS = LSMOD.Nystrom(prob.internal^2,M,Nys_k,Nys_p)
sol_RNYS = LSMOD.solve(t₀, Δt , N, prob, RandNYS; projection_error=projection_error)
pod = LSMOD.POD(prob.internal^2,M,m)
sol_POD = LSMOD.solve(t₀, Δt , N, prob, pod; projection_error=projection_error)
RQR=LSMOD.RandomizedQR(prob.internal^2,M,m)
sol_Rand = LSMOD.solve(t₀, Δt , N, prob, RQR; projection_error=projection_error)
RSVD=LSMOD.RandomizedSVD(prob.internal^2,M,m)
sol_RandSVD = LSMOD.solve(t₀, Δt, N, prob, RSVD; projection_error=projection_error)

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
         "Nys" => (Nys_k,Nys_p)
         )
    )

