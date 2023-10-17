using LinearAlgebra,ForwardDiff, Revise, Plots, InvertedIndices
include("../src/LSMOD.jl")
using .LSMOD

"""
    We consider the following problem:
        ∇⋅(a(x,y,t)∇f(x,y,t)) = rhs(x,y,t) ∀(x,y) ∈ Ω
        f(x,y,t)=0 ∀(x,y) ∈ ∂Ω
    With Ω ⊂ [0,1]²
"""
#Wave speed
a(x::Vector, t) = exp(-(x[1] - 0.5)^2 - (x[2] - 0.5)^2) * cos(x[1] * t) + 2.1
a(x::Tuple, t) = a([x...],t)

#Exact solution
exact(x::Vector, t) = sin(4*pi*x[1])*sin(4*pi*x[2])*(1 + sin(15*pi*x[1]*t)*sin(3*pi*x[2]*t)*exp(-(x[1]-0.5)^2 - (x[2]-0.5)^2 - 0.25^2)) 
exact(x::Tuple, t) = exact([x...],t) 

#We calculate the rhs function using automatic differentiation
function rhs(x::Tuple,t,a,exact)
    frozen_a(y::Vector) = a(y,t)
    frozen_exact(y::Vector) = exact(y,t)
    
    x_vec = [x...]
    d1 = ForwardDiff.gradient(frozen_a,x_vec)' * ForwardDiff.gradient(frozen_exact,x_vec)
    d2 = frozen_a(x_vec) * tr(ForwardDiff.hessian(frozen_exact,x_vec))
    return d1 + d2
end

g(x::Tuple,t) = rhs(x,t,a,exact)

prob = LSMOD.EllipticPDE(
	100, # discretisation (each direction)
	0.0, # xmin
	1.0, # xmax
	0.0, # ymin
	1.0, # ymax
	a, # wave speed (~ish)
	g, # rhs
)

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

sol_base_3,_ = LSMOD.solve(2.3, Δt_3 , N, prob)
pod = LSMOD.POD(prob.internal^2,M_3,m_3)
sol_POD_3 = LSMOD.solve(2.3, Δt_3 , N, prob, pod)
RQR=LSMOD.RandomizedQR(prob.internal^2,M_3,m_3)
sol_Rand_3 = LSMOD.solve(2.3, Δt_3 , N, prob, RQR)
RSVD=LSMOD.RandomizedSVD(prob.internal^2,M_3,m_3)
sol_RandSVD_3 = LSMOD.solve(2.3, Δt_3 , N, prob, RSVD)


extract_iters(v) = [v[el][:history].niter for el in range(2,length(v))]
l1 = extract_iters(sol_base_3)
l2 = extract_iters(sol_POD_3)
l3 = extract_iters(sol_Rand_3)
l4 = extract_iters(sol_RandSVD_3)

fig1 = scatter(steps, 
            [l1,l2,l3,l4], 
            title = "Δt = 10⁻³, M=35, m=20", 
            label= ["Base" "POD" "Randomized QR" "Randomized SVD"], 
            lw=2,
            xlabel="Timestep",
            ylabel="GMRES iterations")
Plots.savefig("Figures/10_3_with_precond_opt.png")

#Δt = 10⁻⁵
M_5=20
m_5=10
Δt_5 = 1e-5

sol_base_5,_ = LSMOD.solve(2.3, Δt_5 , N, prob)
pod_2 = LSMOD.POD(prob.internal^2,M_5,m_5)
sol_POD_5 = LSMOD.solve(2.3, Δt_5 , N, prob, pod_2)
RQR_2 =LSMOD.RandomizedQR(prob.internal^2,M_5,m_5)
sol_Rand_5 = LSMOD.solve(2.3, Δt_5 , N, prob, RQR_2)
RSVD_2 =LSMOD.RandomizedSVD(prob.internal^2,M_5,m_5)
sol_RandSVD_5 = LSMOD.solve(2.3, Δt_5 , N, prob, RSVD_2)


l1_2 = extract_iters(sol_base_5)
l2_2 = extract_iters(sol_POD_5)
l3_2 = extract_iters(sol_Rand_5)
l4_2 = extract_iters(sol_RandSVD_5)


fig2 = scatter(steps, 
            [l1_2,l2_2,l3_2, l4_2], 
            title = "Δt = 10⁻⁵, M=20, m=10", 
            label= ["Base" "POD" "Randomized QR" "Randomized SVD"], 
            xlabel="Timestep",
            ylabel="GMRES iterations")
Plots.savefig("Figures/10_5_with_precond_opt.png")
