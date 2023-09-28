using Plots, Revise, LinearAlgebra
include("../src/LSMOD.jl")

rhs(x, y, t) =
	(1.0 - 2 * y) * (4 * pi * t * sin(4 * pi * t * x) * cos(4 * pi * t * y) + 0.939413062813476 * (2 * y - 1.0) * exp((x - 0.5)^2 + (y - 0.5)^2) * sin(15 * pi * t * x)) * exp(-(x - 0.5)^2 - (y - 0.5)^2) * cos(t * x) +
	(exp(-(x - 0.5)^2 - (y - 0.5)^2) * cos(t * x) + 2.1) *
	(-16 * pi^2 * t^2 * sin(4 * pi * t * x) * sin(4 * pi * t * y) + 0.939413062813476 * (4 * (y - 0.5)^2) * exp((x - 0.5)^2 + (y - 0.5)^2) * sin(15 * pi * t * x) + 1.87882612562695 * exp((x - 0.5)^2 + (y - 0.5)^2) * sin(15 * pi * t * x)) +
	(exp(-(x - 0.5)^2 - (y - 0.5)^2) * cos(t * x) + 2.1) * (
		-211.367939133032 * pi^2 * t^2 * exp((x - 0.5)^2 + (y - 0.5)^2) * sin(15 * pi * t * x) - 16 * pi^2 * t^2 * sin(4 * pi * t * x) * sin(4 * pi * t * y) +
		28.1823918844043 * pi * t * (2 * x - 1.0) * exp((x - 0.5)^2 + (y - 0.5)^2) * cos(15 * pi * t * x) + 0.939413062813476 * (4 * (x - 0.5)^2) * exp((x - 0.5)^2 + (y - 0.5)^2) * sin(15 * pi * t * x) +
		1.87882612562695 * exp((x - 0.5)^2 + (y - 0.5)^2) * sin(15 * pi * t * x)
	) +
	(-t * exp(-(x - 0.5)^2 - (y - 0.5)^2) * sin(t * x) + (1.0 - 2 * x) * exp(-(x - 0.5)^2 - (y - 0.5)^2) * cos(t * x)) *
	(14.0911959422021 * pi * t * exp((x - 0.5)^2 + (y - 0.5)^2) * cos(15 * pi * t * x) + 4 * pi * t * sin(4 * pi * t * y) * cos(4 * pi * t * x) + 0.939413062813476 * (2 * x - 1.0) * exp((x - 0.5)^2 + (y - 0.5)^2) * sin(15 * pi * t * x))

#g(x::Tuple, t) = -sin(t)*2*(pi)^2*sin(pi*x[1])*sin(pi*x[2])
g(x::Tuple, t) = rhs(x[1], x[2], t)

#a(x::Tuple, t) = 1
a(x::Tuple, t) = exp(-(x[1] - 0.5)^2 - (x[2] - 0.5)^2) * cos(x[1] * t) + 2.1

prob = LSMOD.EllipticPDE(
	100, # discretisation (each direction)
	0.0, # xmin
	1.0, # xmax
	0.0, # ymin
	1.0, # ymax
	a, # wave speed (~ish)
	g, # rhs
)
#exact(x, t) = sin(pi*x[1])*sin(pi*x[2])*sin(t)
exact(x, t) = sin(4 * pi * x[2] * t) * sin(4 * pi * x[1] * t) + sin(15 * pi * x[1] * t) * exp((x[1] - 0.5)^2 + (x[2] - 0.5)^2 + 0.25^2)
sols = LSMOD.solve(2.3, 1e-3, 200, prob);

eff(x) = exact(x, sols[end][:time])

# surface(collect(Iterators.flatten(getfield.(prob.grid, 1))),
# 	collect(Iterators.flatten(getfield.(prob.grid, 2))),
# 	sols[end][:x])

surface(collect(Iterators.flatten(getfield.(prob.grid, 1))),
	collect(Iterators.flatten(getfield.(prob.grid, 2))),
	collect(Iterators.flatmap(eff,prob.grid)) - sols[end][:x])

#println(norm(sols[end]["x"] - collect(Iterators.flatmap(eff,prob.grid)))/norm(sols[end]["x"]))

#show(plot)