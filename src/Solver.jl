using IterativeSolvers: IterativeSolvers

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem)
	"""
		Baseline iterative solver
	"""
	dim = size(problem.grid)
	n = dim[1] * dim[2]
	sol = randn(n)
	solutions = Array{Dict}(undef, N)
	for (i, time) ∈ enumerate(t₀:Δt:(t₀+(N-1)*Δt))
		A, rhs = updateLinearSystem(problem, time)
		sol, history = IterativeSolvers.gmres!(sol, A, rhs; log = true, reltol = 1e-7)
		solutions[i] = Dict(:x => sol, :history => history, :time => time)
	end

	return solutions
end
