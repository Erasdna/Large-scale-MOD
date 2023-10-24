export orderReduction!
export Strategy, POD, RandomizedQR, RandomizedSVD, Nystrom

abstract type Strategy end;
abstract type RandomizedStrategy <: Strategy end;

mutable struct POD <: Strategy
	const M::Integer
	const m::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	function POD(dim::Integer, M::Integer, m::Integer)
		return new(
			M,
			m,
			Matrix{Float64}(undef, dim, M), #History: R^(dim × M)
			Matrix{Float64}(undef, dim, m), #reduced basis: R^(dim × m)
		)
	end
end

function orderReduction!(strategy::POD)
	"""
		Calculates SVD of a sample matrix
	"""
	F = svd(strategy.solutions)
	strategy.basis .= @view Matrix(F.U)[:, 1:strategy.m]
end

mutable struct RandomizedQR <: RandomizedStrategy
	const M::Integer
	const m::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	Ω::AbstractMatrix
	old_Ω::AbstractMatrix
	Z::AbstractMatrix
	counter::Integer
	const freq::Integer

	function RandomizedQR(dim::Integer, M::Integer, m::Integer; freq::Integer = 50)
		return new(
			M,
			m,
			Matrix{Float64}(undef, dim, M), #History: R^(dim × M)
			Matrix{Float64}(undef, dim, m), #reduced basis: R^(dim × m)
			Matrix{Float64}(undef, dim, m),
			Matrix{Float64}(undef, dim, m),
			Matrix{Float64}(undef, M, m),
			0,
			freq,
		)
	end
end

function sketch_update!(strategy::RandomizedStrategy)
	"""
		Builds and updates a sketch matrix of the sample
	"""
	if strategy.counter == strategy.freq || strategy.counter == 0
		strategy.Z .= randn((strategy.M, strategy.m))
		#strategy.Ω .= strategy.solutions * strategy.Z
		mul!(strategy.Ω, strategy.solutions, strategy.Z)
		strategy.counter += 1
	else
		z = randn(strategy.m)
		@. strategy.Ω += strategy.solutions[:, end] * (z') - strategy.old_Ω
		#cycle_and_replace!(strategy.Z,z; col=false)
		strategy.Z .= circshift(strategy.Z, (-1, 0))
		strategy.Z[end, :] .= z
	end

	strategy.old_Ω .= strategy.solutions[:, 1] * (strategy.Z[1, :]')
end

function orderReduction!(strategy::RandomizedQR)
	"""
		Computes the randomized QR of a sample matrix
	"""
	sketch_update!(strategy)

	#Check out QRUpdate.jl?
	F = qr(strategy.Ω)
	strategy.basis .= @view Matrix(F.Q)[:, 1:strategy.m]
end

mutable struct RandomizedSVD <: RandomizedStrategy
	const M::Integer
	const m::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	Ω::AbstractMatrix
	B::AbstractMatrix
	Q::AbstractMatrix
	old_Ω::AbstractMatrix
	Z::AbstractMatrix
	counter::Integer
	const freq::Integer

	function RandomizedSVD(dim::Integer, M::Integer, m::Integer; freq::Integer = 50)
		return new(
			M,
			m,
			Matrix{Float64}(undef, dim, M), #History: R^(dim × M)
			Matrix{Float64}(undef, dim, m), #reduced basis: R^(dim × m)
			Matrix{Float64}(undef, dim, m),
			Matrix{Float64}(undef, m, M),
			Matrix{Float64}(undef, dim, m),
			Matrix{Float64}(undef, dim, m),
			Matrix{Float64}(undef, M, m),
			0,
			freq,
		)
	end
end

function orderReduction!(strategy::RandomizedSVD)
	"""
		Computes the randomized SVD of a sample matrix
	"""
	#sketch
	sketch_update!(strategy)

	#QR
	F = qr(strategy.Ω)
	strategy.Q .= Matrix(F.Q)

	#SVD
	mul!(strategy.B, strategy.Q', strategy.solutions)
	Fb = svd(strategy.B)

	strategy.basis .= @view (strategy.Q*Fb.U)[:, 1:strategy.m]
end

mutable struct Nystrom <: RandomizedStrategy
	const M::Integer
	const r::Integer
	const l::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	const Ω₁::AbstractMatrix # left sample matrix 
	const Ω₂::AbstractMatrix # right sample matrix
	prod1::AbstractMatrix
	prod2::AbstractMatrix

	function Nystrom(dim::Integer, M::Integer, r::Integer, l::Integer)
		return new(
			M,
			r,
			l,
			Matrix{Float64}(undef, dim, M),
			Matrix{Float64}(undef, dim, r + l),
			randn(M, r),
			randn(dim, r + l),
			Matrix{Float64}(undef, dim, r),
			Matrix{Float64}(undef, r + l, r),
		)
	end
end

function orderReduction!(strategy::Nystrom)
	"""
		Computes the left side projection operator of the generalized Nystrom approximation
	"""
	# XΩ₁
	mul!(strategy.prod1, strategy.solutions, strategy.Ω₁)
	# Ω₂ᵀXΩ₁
	mul!(strategy.prod2, strategy.Ω₂', strategy.prod1)

	# XΩ₁(Ω₂ᵀXΩ₁)^†
	mul!(strategy.basis, strategy.prod1, pinv(strategy.prod2))
end
