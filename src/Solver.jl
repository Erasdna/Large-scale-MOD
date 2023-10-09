import ProgressBars, Printf, Krylov, InvertedIndices, ILUZero
include("ReductionStrategies.jl")

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem)
	"""
		Baseline iterative solver
	"""
	prevsol=[]
	solutions = Array{Dict}(undef, N)

	LU=nothing

	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1)*Δt))) 
	for (i, time) ∈ iter
		A, rhs = updateLinearSystem(problem, time)

		if i==1
			LU=ILUZero.ilu0(A)
			run_gmres!(nothing,A,rhs,iter,solutions,problem,i,time,LU; use_guess=false)
		else
			ILUZero.ilu0!(LU,A)
			run_gmres!(prevsol[InvertedIndices.Not(problem.edge)],A,rhs,iter,solutions,problem,i,time,LU)
		end
		prevsol = copy(solutions[i][:x])
	end

	return solutions,LU
end

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem,M::Integer, m::Integer, reductionMethod)
	solutions = Array{Dict}(undef, N)
	solutions[1:M],LU = solve(t₀,Δt,M,problem)

	mat = hcat([solutions[i][:x] for i ∈ 1:M]...)
	t₀=solutions[M][:time]

	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1-M)*Δt)))
	for (i,time) ∈ iter
		A, rhs = updateLinearSystem(problem, time)
		ILUZero.ilu0!(LU,A)

		basis=reductionMethod(mat[InvertedIndices.Not(problem.edge),1:end],M,m)
		AQ = A * basis

		IG_small = qr(AQ) \ (rhs) #Go through QR + preconditioner
		IG = basis * IG_small

		run_gmres!(IG,A,rhs,iter,solutions,problem,i+M,time,LU)

		mat=cat(mat,solutions[i+M][:x]; dims=2)
	end

	return solutions
end

function run_gmres!(initial_guess,A,rhs,iter,solutions, problem,index,time, precond;use_guess=true)
	if use_guess
		r0 = norm(rhs - A*initial_guess)
		sv = copy(initial_guess)
		sol, history = Krylov.gmres(A, rhs, initial_guess; M=precond, ldiv=true, atol=1e-7*norm(precond \ rhs))
	else
		r0=0
		sv = nothing
		sol, history = Krylov.gmres(A, rhs; M=precond, ldiv=true, rtol=1e-7)
	end
	res = norm(rhs - A*sol)
	ProgressBars.set_postfix(iter, Iterations=Printf.@sprintf("%d",history.niter), r₀=Printf.@sprintf("%4.3e",r0/norm(rhs)), rₙ=Printf.@sprintf("%4.3e",res/norm(rhs)))
	
	#Could do this in one line
	full_sol = zeros((problem.internal+2)^2)
	full_sol[InvertedIndices.Not(problem.edge)] .= sol

	solutions[index] = Dict(:x => full_sol, :history => history, :time => time, :residual => res, :IG => sv)
end
