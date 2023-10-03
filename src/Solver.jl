import ProgressBars, Printf, IterativeSolvers, InvertedIndices
include("ReductionStrategies.jl")

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem)
	"""
		Baseline iterative solver
	"""
	prevsol=[]
	solutions = Array{Dict}(undef, N)
	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1)*Δt))) 
	for (i, time) ∈ iter
		A, rhs = updateLinearSystem(problem, time)
		if i==1
			sol, history = IterativeSolvers.gmres(A, rhs; log = true, abstol = 1e-7)
		else
			sol, history = IterativeSolvers.gmres!(prevsol,A, rhs; log = true, abstol = 1e-7)
		end			
		res = norm(rhs - A*sol)
		ProgressBars.set_postfix(iter, Iterations=Printf.@sprintf("%d",history.iters))
		prevsol = copy(sol)

		#Could do this in one line
		full_sol = zeros((problem.internal+2)^2)
		full_sol[InvertedIndices.Not(problem.edge)] .= sol
		solutions[i] = Dict(:x => full_sol, :history => history, :time => time, :residual => res)
	end

	return solutions
end

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem,M::Integer, m::Integer, reductionMethod)
	solutions = Array{Dict}(undef, N)
	solutions[1:M] = solve(t₀,Δt,M,problem)
	
	mat = hcat([solutions[i][:x] for i ∈ 1:M]...)
	t₀=solutions[M][:time]
	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1-M)*Δt)))
	for (i,time) ∈ iter
		A, rhs = updateLinearSystem(problem, time)
		basis=reductionMethod(mat[InvertedIndices.Not(problem.edge),1:end],M,m)
		AQ = A*basis
		IG_small = AQ \ rhs #Go through QR
		IG = basis * IG_small
		sv = copy(IG)

		#ProgressBars.println(iter,"Initial guess residual: ", norm(A*IG -rhs))
		#incomplete lu
		sol, history = IterativeSolvers.gmres!(IG,A, rhs; log = true, abstol=1e-7, verbose=false)

		res = norm(rhs - A*sol)
		ProgressBars.set_postfix(iter, Iterations=Printf.@sprintf("%d",history.iters), r₀=Printf.@sprintf("%4.3e",res))
		
		#Could do this in one line
		full_sol = zeros((problem.internal+2)^2)
		full_sol[InvertedIndices.Not(problem.edge)] .= sol
		mat=cat(mat,full_sol; dims=2)

		solutions[i+M] = Dict(:x => full_sol, :history => history, :time => time, :residual => res, :IG => sv)
	end

	return solutions
end