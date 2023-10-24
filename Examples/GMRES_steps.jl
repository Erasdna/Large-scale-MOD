using LinearAlgebra,ForwardDiff, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools
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

N=200
steps = 1:200

#Δt = 10⁻³
M_3=35
m_3=20
Δt_3 = 1e-3
t₀=0.0

prob = LSMOD.Example1.prob

sol_base_3,_ = LSMOD.solve(t₀, Δt_3 , N, prob)
RandNYS = LSMOD.Nystrom(prob.internal^2,M_3,14,7)
sol_RNYS_3 = LSMOD.solve(t₀, Δt_3 , N, prob, RandNYS; projection_error=true)
pod = LSMOD.POD(prob.internal^2,M_3,m_3)
sol_POD_3 = LSMOD.solve(t₀, Δt_3 , N, prob, pod; projection_error=true)
RQR=LSMOD.RandomizedQR(prob.internal^2,M_3,m_3)
sol_Rand_3 = LSMOD.solve(t₀, Δt_3 , N, prob, RQR; projection_error=true)
RSVD=LSMOD.RandomizedSVD(prob.internal^2,M_3,m_3)
sol_RandSVD_3 = LSMOD.solve(t₀, Δt_3 , N, prob, RSVD; projection_error=true)


extract_iters(v) = [v[el][:history].niter for el in range(M_3+1,length(v))]
l1 = extract_iters(sol_base_3)
l2 = extract_iters(sol_POD_3)
l3 = extract_iters(sol_Rand_3)
l4 = extract_iters(sol_RandSVD_3)
l5 = extract_iters(sol_RNYS_3)

extract_proj(v) = [v[el][:proj] for el in range(M_3+1,length(v))]
extract_proj_X(v) = [v[el][:proj_X] for el in range(M_3+1,length(v))]
extract_r0(v) = [v[el][:r0] for el in range(M_3+1,length(v))]


fig1 = scatter(range(2,N), 
            [l1,l2,l3,l4,l5], 
            title = L"Δt = 10^{-3}, M=35, m=20", 
            label= ["Base" "POD" "Randomized QR" "Randomized SVD" "Nystrom"], 
            lw=2,
            xlabel="Timestep",
            ylabel="GMRES iterations")
Plots.savefig(fig1,"Figures/10_3_with_precond_opt.png")

proj_X = extract_proj_X(sol_POD_3)
l2_proj = extract_proj(sol_POD_3)
l3_proj = extract_proj(sol_Rand_3)
l4_proj = extract_proj(sol_RandSVD_3)
l5_proj = extract_proj(sol_RNYS_3)


fig3 = scatter(range(M_3+1,N), 
            [proj_X,l2_proj,l3_proj,l4_proj,l5_proj], 
            title = L"Δt = 10^{-3}, M=35, m=20", 
            label= ["QR(X)" "POD" "Randomized QR" "Randomized SVD" "Nystrom"], 
            lw=2,
            xlabel="Timestep",
            ylabel=L"||(I-QQ^T)X||")
Plots.savefig(fig3,"Figures/projection_error.png")

l1_r0 = extract_r0(sol_base_3)
l2_r0 = extract_r0(sol_POD_3)
l3_r0 = extract_r0(sol_Rand_3)
l4_r0 = extract_r0(sol_RandSVD_3)

fig4 = scatter([l2,l3,l4], 
            [l2_r0,l3_r0,l4_r0], 
            title = L"Δt = 10^{-3}, M=35, m=20", 
            label= ["POD" "Randomized QR" "Randomized SVD"], 
            lw=2,
            xlabel="GMRES iterations",
            ylabel=L"||r_0||/||b||")
Plots.savefig(fig4,"Figures/r0_vs_it.png")

#Δt = 10⁻⁵
M_5=20
m_5=10
Δt_5 = 1e-5

sol_base_5,_ = LSMOD.solve(t₀, Δt_5 , N, prob)
pod_2 = LSMOD.POD(prob.internal^2,M_5,m_5)
sol_POD_5 = LSMOD.solve(t₀, Δt_5 , N, prob, pod_2)
RQR_2 =LSMOD.RandomizedQR(prob.internal^2,M_5,m_5)
sol_Rand_5 = LSMOD.solve(t₀, Δt_5 , N, prob, RQR_2)
RSVD_2 =LSMOD.RandomizedSVD(prob.internal^2,M_5,m_5)
sol_RandSVD_5 = LSMOD.solve(t₀, Δt_5 , N, prob, RSVD_2)


l1_2 = extract_iters(sol_base_5)
l2_2 = extract_iters(sol_POD_5)
l3_2 = extract_iters(sol_Rand_5)
l4_2 = extract_iters(sol_RandSVD_5)


fig2 = scatter(steps, 
            [l1_2,l2_2,l3_2, l4_2], 
            title = L"Δt = 10^{-5}, M=20, m=10", 
            label= ["Base" "POD" "Randomized QR" "Randomized SVD"], 
            xlabel="Timestep",
            ylabel="GMRES iterations")
Plots.savefig("Figures/10_5_with_precond_opt.png")
