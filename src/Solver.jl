import ProgressBars, Printf, Krylov, InvertedIndices, ILUZero, IterativeSolvers
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
		preconditioner_time = @elapsed LU=ILUZero.ilu0(A)
		if i==1
			sol, GMRES_time =run_gmres!(nothing,A,rhs,iter,solutions,problem,i,time,LU; use_guess=false)
		else
			sol,GMRES_time=run_gmres!(prevsol,A,rhs,iter,solutions,problem,i,time,LU)
		end
		mat[:,i] .= sol
		prevsol = sol
		solutions[i][:timing] = Dict(:preconditioner => preconditioner_time, :gmres => GMRES_time)
	end

	return solutions,LU,mat
end

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem, strategy::Strategy; projection_error=false)
	solutions = Array{Dict}(undef, N)
	solutions[1:strategy.M],LU,mat = solve(t₀,Δt,strategy.M,problem)
	strategy.solutions .= mat # Need more general strategy!

	t₀=solutions[strategy.M][:time]
	
	A = spzeros(problem.internal^2,problem.internal^2)
	rhs = zeros(problem.internal^2)

	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1-strategy.M)*Δt)))
	for (i,time) ∈ iter
		updateLinearSystem!(A,rhs,problem, time)
		preconditioner_time = @elapsed ILUZero.ilu0!(LU,A)

		basis_time = @elapsed orderReduction!(strategy)
		
		#ProgressBars.println(iter," ||(I - QQᵀ)X||₂ ", norm((I(problem.internal^2) - basis * basis' )*mat))
		guess_time = @elapsed IG,guess_timing = generate_guess(A,strategy.basis,rhs)
		sol,GMRES_time =run_gmres!(IG,A,rhs,iter,solutions,problem,i+strategy.M,time,LU)
		
		if projection_error
			#ProgressBars.println(iter,norm(strategy.solutions - strategy.basis*strategy.basis' * strategy.solutions)/norm(strategy.solutions))
			solutions[i+strategy.M][:proj] = norm(strategy.solutions - strategy.basis*strategy.basis' * strategy.solutions)/norm(strategy.solutions)
			Ff = qr(strategy.solutions)
			Qq = Matrix(Ff.Q)
			solutions[i+strategy.M][:proj_X] = norm(strategy.solutions - Qq*Qq' * strategy.solutions)/norm(strategy.solutions)
		end

		#cycle_and_replace!(strategy.solutions,sol)
		strategy.solutions .= circshift(strategy.solutions,(0,-1))
		strategy.solutions[:,end] .= sol

		solutions[i+strategy.M][:timing] = Dict(:preconditioner => preconditioner_time, :gmres => GMRES_time, :basis => basis_time, :guess => guess_time, :guess_detailed => guess_timing)
	end

	return solutions
end

function generate_guess(A,basis,rhs)
	AQ_time = @elapsed AQ = A * basis

	IG_small=copy(rhs)
	LS_time = @elapsed _,ig,_=LAPACK.gels!('N',AQ,IG_small) 
	IG_time = @elapsed IG = basis * ig

	timing = Dict(:AQ => AQ_time, :LS => LS_time, :IG => IG_time)
	return IG,timing
end

function run_gmres!(initial_guess,A,rhs,iter,solutions, problem,index,time, precond;use_guess=true)
	if use_guess
		r0 = norm(rhs - A*initial_guess)
		sv = copy(initial_guess)
		GMRES_time = @elapsed sol, history = Krylov.gmres(A, rhs, initial_guess; N=precond, ldiv=true, atol=1e-7*norm(rhs))
	else
		r0=0
		sv = nothing
		GMRES_time = @elapsed sol, history = Krylov.gmres(A, rhs; N=precond, ldiv=true, atol=1e-7*norm(rhs))
	end
	res = norm(rhs - A*sol)
	ProgressBars.set_postfix(iter, Iterations=Printf.@sprintf("%d",history.niter), r₀=Printf.@sprintf("%4.3e",r0/norm(rhs)), rₙ=Printf.@sprintf("%4.3e",res/norm(rhs)))
	
	#Could do this in one line
	full_sol = zeros((problem.internal+2)^2)
	full_sol[InvertedIndices.Not(problem.edge)] .= sol

	solutions[index] = Dict(:x => full_sol, :history => history, :time => time, :residual => res, :IG => sv, :r0 => r0/norm(rhs))
	return sol, GMRES_time
end
