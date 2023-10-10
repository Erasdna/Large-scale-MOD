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
sol_base_3,_ = LSMOD.solve(2.3, 1e-3, N, prob)
sol_POD_3 = LSMOD.solve(2.3, 1e-3, N, prob, 20,10,LSMOD.POD!)
sol_Rand_3 = LSMOD.solve(2.3, 1e-3, N, prob, 20,10,LSMOD.RandomizedQR!)

extract_iters(v) = [v[el][:history].niter for el in eachindex(v)]
l1 = extract_iters(sol_base_3)
l2 = extract_iters(sol_POD_3)
l3 = extract_iters(sol_Rand_3)
fig1 = scatter(steps, 
            [l1,l2,l3], 
            title = "Δt = 10⁻³, M=20, m=10", 
            label= ["Base" "POD" "Randomized QR"], 
            lw=2,
            xlabel="Timestep",
            ylabel="GMRES iterations")
Plots.savefig("Figures/10_3_with_precond.png")

#Δt = 10⁻⁵
sol_base_5,_ = LSMOD.solve(2.3, 1e-5, N, prob)
sol_POD_5 = LSMOD.solve(2.3, 1e-5, N, prob, 35,20,LSMOD.POD!)
sol_Rand_5 = LSMOD.solve(2.3, 1e-5, N, prob, 35,20,LSMOD.RandomizedQR!)

l1_2 = extract_iters(sol_base_5)
l2_2 = extract_iters(sol_POD_5)
l3_2 = extract_iters(sol_Rand_5)

fig2 = scatter(steps, 
            [l1_2,l2_2,l3_2], 
            title = "Δt = 10⁻⁵, M=35, m=20", 
            label= ["Base" "POD" "Randomized QR"], 
            xlabel="Timestep",
            ylabel="GMRES iterations")
Plots.savefig("Figures/10_5_with_precond.png")
