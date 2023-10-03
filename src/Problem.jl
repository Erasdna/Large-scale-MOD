export Problem, EllipticPDE
export DifferentialOperators2D

abstract type DifferentialOperators end
abstract type Problem end

struct DifferentialOperators2D
	"""
		2D differential operators
	"""
	∂x::SparseMatrixCSC
	∂y::SparseMatrixCSC
	∂²x::SparseMatrixCSC
	∂²y::SparseMatrixCSC
	function DifferentialOperators2D(N::Int64, hx::Float64, hy::Float64)
		#Fourth order discretisation

		D = spdiagm(
			-2 => (-1) .* ones(N - 2),
			-1 => (8) .* ones(N - 1),
			2 => ones(N - 2),
			1 => (-8) .* ones(N - 1),
		)

		DD = spdiagm(
			-2 => (-1) .* ones(N - 2),
			-1 => (16) .* ones(N - 1),
			0 => (-30) .* ones(N),
			2 => (-1) .* ones(N - 2),
			1 => (16) .* ones(N - 1),
		)
		
		id = sparse(I, N, N)
		new(kron(id, D) ./ (12 * hx), kron(D, id) ./ (12 * hy), kron(id, DD) ./ (12 * hx^2), kron(DD, id) ./ (12 * hy^2))
	end

end

struct EllipticPDE <: Problem
	"""
		Problem on the form ∇⋅(a(x,t)∇f(x,t)) = rhs(x,t)
		with corresponding grid and differential operators
	"""
	internal::Integer
	grid::Any
	a::Any
	rhs::Any
	∂D::DifferentialOperators2D
	edge
	function EllipticPDE(
		in::Int64,
		xmin::Float64,
		xmax::Float64,
		ymin::Float64,
		ymax::Float64,
		a,
		rhs,
	)
		@assert ymax > ymin
		@assert xmax > xmin

		N = in+2
		new(in,collect(Iterators.product(range(xmin, xmax, N), range(ymin, ymax, N))), a, rhs, DifferentialOperators2D(in, (xmax - xmin) / N, (ymax - ymin) / N),sort(collect([1:N; (N+1):N:(N^2 - 2*N + 1); (2*N):N:(N^2 -N); (N^2 - N + 1):N^2])))
	end
end


