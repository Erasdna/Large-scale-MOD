using LinearAlgebra,ForwardDiff, Revise, Plots
include("../src/LSMOD.jl")
using .LSMOD

prob = LSMOD.Example1.make_prob(100)

N=1000
M=20
m=10

Nys_k = 7
Nys_p = 3
Δt = 1e-5
t₀=0.1

sols,_ = LSMOD.solve(t₀, Δt, N, prob);
#RandNYS = LSMOD.Nystrom(prob.internal^2,M,Nys_k,Nys_p)
#sols = LSMOD.solve(t₀, Δt , N, prob, RandNYS)

eff(x) = LSMOD.Example1.exact1(x, sols[end][:time])

println(norm(collect(Iterators.flatmap(eff,prob.grid)) - sols[end][:x])/norm(sols[end][:x]))
ff = surface(collect(Iterators.flatten(getfield.(prob.grid, 1))),
	collect(Iterators.flatten(getfield.(prob.grid, 2))),
	sols[end][:x]);

ff2 = surface(collect(Iterators.flatten(getfield.(prob.grid, 1))),
	collect(Iterators.flatten(getfield.(prob.grid, 2))),
	abs.(collect(Iterators.flatmap(eff,prob.grid)) -sols[end][:x]));

