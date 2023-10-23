using LinearAlgebra,ForwardDiff, Revise, Plots, InvertedIndices
include("../src/LSMOD.jl")
using .LSMOD

function extract_timing(data,start)
    precon = [data[el][:timing][:preconditioner] for el in range(start,length(data))]
    basis = [haskey(data[el][:timing],:basis) ? data[el][:timing][:basis] : 0.0 for el in range(start,length(data))]
    guess = [haskey(data[el][:timing],:guess) ? data[el][:timing][:guess] : 0.0 for el in range(start,length(data))]
    gmres = [data[el][:timing][:gmres] for el in range(start,length(data))]
    non_gmres = basis + guess
    return precon,basis,guess,gmres, non_gmres
end

function extract_reduced_timing(data,start)
    AQ = [data[el][:timing][:guess_detailed][:AQ] for el in range(start,length(data))]
    LS = [data[el][:timing][:guess_detailed][:LS] for el in range(start,length(data))]
    IG = [data[el][:timing][:guess_detailed][:IG] for el in range(start,length(data))]
    return AQ,LS,IG
end

function solver_time(data,M::Integer, title::String)
    precon,basis,guess,gmres,non_gmres = extract_timing(data,M)

    #Proportion of each timing
    fig1 = scatter(M:M+length(precon), 
            [cumsum(precon),cumsum(basis),cumsum(guess),cumsum(gmres)], 
            title = title, 
            label= ["Preconditioner" "Basis" "Guess" "GMRES"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Cumulative time [s]")
    fig2 = scatter(M:M+length(precon), 
            [non_gmres,gmres], 
            title = title, 
            label= ["Initial guess" "GMRES"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Cumulative time [s]")
    return fig1,fig2
end

function method_comp(base,method1, tag1::String, method2, tag2::String,M)
    b = extract_timing(base,M)
    m1 = extract_timing(method1,M)
    m2 = extract_timing(method2,M)
    fig1 = scatter(M:M+length(b[1]), 
            [b[end] + b[end-1],m1[end] + m1[end-1],m2[end] + m2[end-1]], 
            label= ["base" tag1 tag2], 
            lw=2,
            xlabel="Timestep",
            ylabel="Time [s]")
    fig2 = scatter(M:M+length(b[1]), 
            [cumsum(b[end] + b[end-1]),cumsum(m1[end] + m1[end-1]),cumsum(m2[end] + m2[end-1])], 
            label= ["base" tag1 tag2], 
            lw=2,
            xlabel="Timestep",
            ylabel="Time [s]")
    return fig1,fig2
end

function reduced_time(data,M::Integer, title::String)
    AQ,LS,IG = extract_reduced_timing(data,M)

    #Proportion of each timing
    fig1 = scatter(M:M+length(AQ), 
            [AQ,LS,IG], 
            title = title, 
            label= ["AQ" "Least squares" "Qs"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Time [s]")
    return fig1
end
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
            - Fourth order discretisation scheme in space
"""

N=200
steps = 1:200
M_3=35
m_3=20
t₀=0.0
Δt=1e-3

#Δt = 10⁻³
sol_base_3,_ = LSMOD.solve(t₀, Δt, N, prob)
pod = LSMOD.POD(prob.internal^2,M_3,m_3)
sol_POD_3 = LSMOD.solve(t₀, Δt, N, prob, pod)
RQR=LSMOD.RandomizedQR(prob.internal^2,M_3,m_3)
sol_Rand_3 = LSMOD.solve(t₀, Δt, N, prob, RQR)

start = 40 
base1,base2 = solver_time(sol_base_3,start,"base")
savefig(base1, "Figures/Timing/base1.png")
savefig(base2, "Figures/Timing/base2.png")

POD1,POD2 = solver_time(sol_POD_3,start,"POD")
savefig(POD1, "Figures/Timing/POD1.png")
savefig(POD2, "Figures/Timing/POD2.png")

POD3 = reduced_time(sol_POD_3,start,"POD")
savefig(POD3, "Figures/Timing/POD3.png")

Randomized1,Randomized2 = solver_time(sol_Rand_3,start,"Randomized QR")
savefig(Randomized1, "Figures/Timing/Randomized1.png")
savefig(Randomized2, "Figures/Timing/Randomized2.png")

Randomized3 = reduced_time(sol_Rand_3,start,"Randomized QR")
savefig(Randomized3, "Figures/Timing/Randomized3.png")

f3,f4 = method_comp(sol_base_3,sol_POD_3,"POD",sol_Rand_3,"Randomized",start)
savefig(f3, "Figures/Timing/comp.png")
savefig(f4, "Figures/Timing/Cumulative_comp.png")
