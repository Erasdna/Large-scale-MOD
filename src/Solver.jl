import ProgressBars, Printf, IterativeSolvers

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem)
	"""
		Baseline iterative solver
	"""
	solutions = Array{Dict}(undef, N)
	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1)*Δt))) 
	for (i, time) ∈ iter
		A, rhs = updateLinearSystem(problem, time)
		if i==1
			sol, history = IterativeSolvers.gmres(A, rhs; log = true, reltol = 1e-7)
		else
			sol, history = IterativeSolvers.gmres!(prevsol,A, rhs; log = true, reltol = 1e-7)
		end			
		res = norm(rhs - A*sol)
		ProgressBars.set_postfix(iter, Resdiual=Printf.@sprintf("%.5f", res))
		prevsol = copy(sol)
		solutions[i] = Dict(:x => sol, :history => history, :time => time, :residual => res)
	end

	return solutions
end
