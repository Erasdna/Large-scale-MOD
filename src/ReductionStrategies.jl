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
	const p::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	Ω::AbstractMatrix
	old_Ω::AbstractMatrix
	Z::AbstractMatrix
	counter::Integer
	const freq::Integer

	function RandomizedQR(dim::Integer, M::Integer, m::Integer; freq::Integer = 50, p::Integer=0)
		return new(
			M,
			m,
			p,
			Matrix{Float64}(undef, dim, M), #History: R^(dim × M)
			Matrix{Float64}(undef, dim, m), #reduced basis: R^(dim × m)
			Matrix{Float64}(undef, dim, m+p),
			Matrix{Float64}(undef, dim, m+p),
			Matrix{Float64}(undef, M, m+p),
			0,
			freq,
		)
	end
end

function sketch_update!(strategy::RandomizedStrategy)
	"""
		Builds and updates a sketch matrix of the sample
	"""
	if strategy.counter % strategy.freq == 0 || strategy.counter == 0
		strategy.Z .= randn((strategy.M, strategy.m + strategy.p))
		mul!(strategy.Ω, strategy.solutions, strategy.Z)
		strategy.counter += 1
	else
		z = randn(strategy.m+strategy.p)
		@. strategy.Ω += strategy.solutions[:, end] * (z') - strategy.old_Ω
		strategy.Z .= circshift(strategy.Z, (-1, 0))
		strategy.Z[end, :] .= z
		strategy.counter+=1
	end

	strategy.old_Ω .= strategy.solutions[:, 1] * (strategy.Z[1, :]')
end

function orderReduction!(strategy::RandomizedQR)
	"""
		Computes the randomized QR (Range Finder) of a sample matrix
	"""
	#Updates the sketch
	sketch_update!(strategy)

	#Computes QR with LAPACK
	tmp,tau=LAPACK.geqrf!(copy(strategy.Ω))
	if strategy.p !=0
		R = @view triu(tmp)[1:strategy.m + strategy.p ,1:strategy.m + strategy.p]
		fac = svd(R)
		LAPACK.orgqr!(tmp,tau)
		strategy.basis .= tmp * fac.U[:,1:strategy.m]
	else
		LAPACK.orgqr!(tmp,tau)
		strategy.basis .= @view tmp[:,1:strategy.m]
	end
	
end

mutable struct RandomizedSVD <: RandomizedStrategy
	const M::Integer
	const p::Integer
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

	function RandomizedSVD(dim::Integer, M::Integer, m::Integer; freq::Integer = 50, p::Integer=0)
		return new(
			M,
			p,
			m,
			Matrix{Float64}(undef, dim, M), #History: R^(dim × M)
			Matrix{Float64}(undef, dim, m), #reduced basis: R^(dim × m)
			Matrix{Float64}(undef, dim, m+p),
			Matrix{Float64}(undef, m+p, M),
			Matrix{Float64}(undef, dim, m+p),
			Matrix{Float64}(undef, dim, m+p),
			Matrix{Float64}(undef, M, m+p),
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
	strategy.Q,tau=LAPACK.geqrf!(copy(strategy.Ω))
	LAPACK.orgqr!(strategy.Q,tau)

	#SVD
	mul!(strategy.B, strategy.Q', strategy.solutions)
	LAPACK.gesvd!('O','N',strategy.B)
	mat = @view strategy.B[:,1:strategy.m]
	strategy.basis .= strategy.Q*mat
end

mutable struct Nystrom <: RandomizedStrategy
	const M::Integer
	const k::Integer
	const p::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	const Ω₁::AbstractMatrix # left sample matrix 
	const Ω₂::AbstractMatrix # right sample matrix
	prod1::AbstractMatrix
	prod2::AbstractMatrix
	R::AbstractMatrix
	const dim :: Integer

	function Nystrom(dim::Integer, M::Integer, k::Integer, p::Integer)
		return new(
			M,
			k,
			p,
			Matrix{Float64}(undef, dim, M),
			Matrix{Float64}(undef, dim, k + p),
			randn(M, k),
			randn(dim, k + p),
			Matrix{Float64}(undef, dim, k),
			Matrix{Float64}(undef, k + p, k),
			Matrix{Float64}(undef, k, k),
			dim,
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
	_,tau=LAPACK.geqrf!(strategy.prod2)
	strategy.R .= @view triu(strategy.prod2)[1:strategy.k,1:strategy.k]
	LAPACK.orgqr!(strategy.prod2,tau)

	mul!(strategy.basis,strategy.prod1 / strategy.R,strategy.prod2')
end
