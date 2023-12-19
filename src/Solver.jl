using ProgressBars: ProgressBars
using Printf: Printf
using Krylov: Krylov
using InvertedIndices: InvertedIndices
using ILUZero: ILUZero

function solve(t₀::Number, Δt::Number, N::Integer, problem::Problem, args...)
	"""
		Baseline iterative solver

		Input:
			- t₀ (Number): Initial time of the simulation
			- Δt (Number): Timestep of the simulation
			- N (Integer): Number of timesteps
			- problem (Problem): problem to be solved
			- args..
		output:
			- solutions (Array{Dict}): Array of dictionaries containing information about the results of the simulation
			- LU (Matrix): Incomplete LU preconditioner at the last timestep
			- mat (Matrix): Matrix of the result states  
	"""
	prevsol = []
	solutions = Array{Dict}(undef, N)

	LU = nothing

	mat = Matrix{Float64}(undef, problem.internal^2, N)

	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1)*Δt)))
	for (i, time) ∈ iter
		updateLinearSystem!(problem, time)
		preconditioner_time = @elapsed LU = ILUZero.ilu0(problem.update.A)
		if i == 1
			sol, GMRES_time = run_gmres!(nothing, problem.update.A, problem.update.rhs_vec, iter, solutions, problem, i, time, LU; use_guess = false)
		else
			sol, GMRES_time = run_gmres!(prevsol, problem.update.A, problem.update.rhs_vec, iter, solutions, problem, i, time, LU)
		end
		mat[:, i] .= sol
		prevsol = sol
		solutions[i][:timing] = Dict(:preconditioner => preconditioner_time, :gmres => GMRES_time)
	end

	return solutions, LU, mat
end

function solve(t₀::Number, Δt::Number, N::Int64, problem::Problem, strategy::Strategy, LS = nothing; projection_error = false)
	"""
		Iterative solver using an initial guess strategy and optionally a randomized least squares method

		Input:
			- t₀ (Number): Initial time of the simulation
			- Δt (Number): Timestep of the simulation
			- N (Integer): Number of timesteps
			- problem (Problem): problem to be solved
			- strategy (Strategy): Initial guess strategy
			- LS (Function) Optional : Randomized least squares method
			- projection_error (Bool): Bool deciding whether or not the projection error of the approximation is calculated
		output:
			- solutions (Array{Dict}): Array of dictionaries containing information about the results of the simulation
	"""
	solutions = Array{Dict}(undef, N)
	solutions[1:strategy.M], LU, mat = solve(t₀, Δt, strategy.M, problem)
	strategy.solutions .= mat # Need more general strategy!

	t₀ = solutions[strategy.M][:time]

	iter = ProgressBars.ProgressBar(enumerate(t₀:Δt:(t₀+(N-1-strategy.M)*Δt)))
	for (i, time) ∈ iter
		updateLinearSystem!(problem, time)
		preconditioner_time = @elapsed ILUZero.ilu0!(LU, problem.update.A)

		basis_time = @elapsed orderReduction!(strategy)

		if projection_error
			tmp = proj(strategy)
		end
		
		#Generate initial guess 
		guess_time = @elapsed IG, guess_timing = generate_guess(strategy.basis, problem, time; LS = LS)
		#Run GMRES
		sol, GMRES_time = run_gmres!(IG, problem.update.A, problem.update.rhs_vec, iter, solutions, problem, i + strategy.M, time, LU)
		#Calculate the projection errors
		if projection_error
			solutions[i+strategy.M][:proj] = tmp
		end

		#Update history
		strategy.solutions = circshift(strategy.solutions, (0, -1))
		strategy.solutions[:, end] .= sol

		solutions[i+strategy.M][:timing] = Dict(:preconditioner => preconditioner_time, :gmres => GMRES_time, :basis => basis_time, :guess => guess_time, :guess_detailed => guess_timing)
	end

	return solutions
end

function generate_guess(basis::Matrix, problem::Problem, time::Number; LS = nothing)
	"""
		Generates an initial guess for GMRES with the help of a precomputed basis 
		
		Input:
			- basis (Matrix): Reduced order Matrix
			- problem (Problem): struct containing information about the problem
			- time (Number): The current time 
			- LS (Optional): Applies a reduction strategy to the initial guess least squares problem
		Output:
			- IG (Vector): Initial guess 
			- timing (Dict): Detailed timing of the initial guess generation
	"""

	if isnothing(LS)
		rhs = copy(problem.update.rhs_vec)
		A = problem.update.A
		red_time = 0
	else
		red_time = @elapsed ind, A, rhs = LS(problem.update.A, problem.update.rhs_vec, problem, time)
	end

	#Create reduced representation of A
	AQ_time = @elapsed AQ = A * basis

	#Solve reduced LS problem  min ||AQs - b||
	LS_time = @elapsed _, ig, _ = LAPACK.gels!('N', AQ, rhs)

	#Recover full representation 
	IG_time = @elapsed IG = basis * ig

	timing = Dict(:AQ => AQ_time, :LS => LS_time, :IG => IG_time, :red => red_time)
	return IG, timing
end

function run_gmres!(initial_guess, A, rhs, iter, solutions, problem, index, time, precond; use_guess = true)
	"""
		Wrapper method for running gmres
	"""
	if use_guess
		r0 = norm(rhs - A * initial_guess)
		sv = copy(initial_guess)
		GMRES_time = @elapsed sol, history = Krylov.gmres(A, rhs, initial_guess; N = precond, ldiv = true, atol = 1e-7 * norm(rhs))
	else
		r0 = 0
		sv = nothing
		GMRES_time = @elapsed sol, history = Krylov.gmres(A, rhs; N = precond, ldiv = true, atol = 1e-7 * norm(rhs))
	end
	res = norm(rhs - A * sol)
	ProgressBars.set_postfix(iter, Iterations = Printf.@sprintf("%d", history.niter), r₀ = Printf.@sprintf("%4.3e", r0 / norm(rhs)), rₙ = Printf.@sprintf("%4.3e", res / norm(rhs)))

	#Pad the solution by adding 0 boundary conditions
	full_sol = zeros((problem.internal + 2)^2)
	full_sol[InvertedIndices.Not(problem.edge)] .= sol

	solutions[index] = Dict(:x => full_sol, :history => history, :time => time, :residual => res, :IG => sv, :r0 => r0 / norm(rhs))
	return sol, GMRES_time
end

function proj(strategy::Strategy)
	"""
		Orthogonal basis projection error
	"""
	return norm(strategy.solutions - strategy.basis * strategy.basis' * strategy.solutions) / norm(strategy.solutions)
end

function proj(strategy::Nystrom)
	"""
		Nyström projection error
	"""
	A_r = strategy.basis * (strategy.Ω₂' * strategy.solutions)
	return norm(strategy.solutions - A_r) / norm(strategy.solutions)
end
