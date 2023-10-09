using LinearAlgebra,ForwardDiff, Revise, Plots, InvertedIndices
include("../src/LSMOD.jl")
using .LSMOD

#Wave speed
a(x::Vector, t) = exp(-(x[1] - 0.5)^2 - (x[2] - 0.5)^2) * cos(x[1] * t) + 2.1
a(x::Tuple, t) = a([x...],t)

#Exact solution
exact(x::Vector, t) = sin(4*pi*x[1])*sin(4*pi*x[2])*(1 + sin(15*pi*x[1]*t)*sin(3*pi*x[2]*t)*exp(-(x[1]-0.5)^2 - (x[2]-0.5)^2 - 0.25^2)) 
exact(x::Tuple, t) = exact([x...],t) #sin(t)*sin(2*pi*x[1])*sin(2*pi*x[2])*(1 + exp((x[1]-0.5)^2 + (x[2]-0.5)^2))#sin(4 * pi * x[2]) * sin(4 * pi * x[1])*(1 + sin(t) * exp((x[1] - 0.5)^2 + (x[2] - 0.5)^2 + 0.25^2))

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

sols = LSMOD.solve(2.3, 1e-5, 25, prob,20,10,LSMOD.POD!);

eff(x) = exact(x, sols[end][:time])

exact_vals = collect(Iterators.flatmap(eff,prob.grid))[Not(prob.edge)]
sol_vals = sols[end][:x][Not(prob.edge)]
println(norm(exact_vals - sol_vals)/norm(sol_vals))

ff2 = surface(collect(Iterators.flatten(getfield.(prob.grid, 1)))[Not(prob.edge)],
	collect(Iterators.flatten(getfield.(prob.grid, 2)))[Not(prob.edge)],
	abs.(exact_vals - sol_vals));

ff3 = surface(collect(Iterators.flatten(getfield.(prob.grid, 1))),
	collect(Iterators.flatten(getfield.(prob.grid, 2))),
	sols[end][:x]);