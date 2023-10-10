import ProgressBars, Printf, Krylov, InvertedIndices, ILUZero
include("ReductionStrategies.jl")

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem)
	"""
		Baseline iterative solver
	"""
	prevsol=[]
	solutions = Array{Dict}(undef, N)

	LU=nothing

	A = spzeros(problem.internal^2,problem.internal^2)
	rhs = zeros(problem.internal^2)
	mat = Matrix{Float64}(undef,problem.internal^2,N)
	
	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1)*Δt))) 
	for (i, time) ∈ iter
		updateLinearSystem!(A,rhs,problem, time)

		if i==1
			LU=ILUZero.ilu0(A)
			sol=run_gmres!(nothing,A,rhs,iter,solutions,problem,i,time,LU; use_guess=false)
		else
			ILUZero.ilu0!(LU,A)
			sol=run_gmres!(prevsol,A,rhs,iter,solutions,problem,i,time,LU)
		end
		mat[:,i] .= copy(sol)
		prevsol = copy(sol)
	end

	return solutions,LU,mat
end

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem, M::Integer, m::Integer, reductionMethod)
	solutions = Array{Dict}(undef, N)
	solutions[1:M],LU,mat = solve(t₀,Δt,M,problem)

	t₀=solutions[M][:time]
	
	A = spzeros(problem.internal^2,problem.internal^2)
	rhs = zeros(problem.internal^2)
	basis = zeros(problem.internal^2,m)

	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1-M)*Δt)))
	for (i,time) ∈ iter
		updateLinearSystem!(A,rhs,problem, time)
		ILUZero.ilu0!(LU,A)

		reductionMethod(basis,mat,M,m)
		# AQ = A * basis

		# IG_small = qr(AQ) \ (rhs) #Go through QR + preconditioner
		# IG = basis * IG_small
		IG = generate_guess(A,basis,rhs)
		sol=run_gmres!(IG,A,rhs,iter,solutions,problem,i+M,time,LU)
		
		#mat=cat(mat,sol; dims=2)
		mat=circshift(mat,(0,1))
		mat[:,end] = sol
	end

	return solutions
end

function generate_guess(A,basis,rhs)
	AQ = A * basis

	IG_small = qr(AQ) \ (rhs) 
	IG = basis * IG_small

	return IG
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
	return sol
end
