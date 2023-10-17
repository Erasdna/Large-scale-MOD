export orderReduction!

export Strategy, POD, RandomizedQR, RandomizedSVD

abstract type Strategy end
abstract type RandomizedStrategy <: Strategy end

mutable struct POD <: Strategy
	M::Integer
	m::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	function POD(dim::Integer,M::Integer,m::Integer)
		return new(
			M,
			m,
			Matrix{Float64}(undef,dim,M), #History: R^(dim × M)
			Matrix{Float64}(undef,dim,m) #reduced basis: R^(dim × m)
		)
	end
end

function orderReduction!(strategy::POD)
	F = svd(strategy.solutions)
	strategy.basis .= @view Matrix(F.U)[:, 1:strategy.m];
end

mutable struct RandomizedQR <: RandomizedStrategy
	M::Integer
	m::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	Ω::AbstractMatrix
	old_Ω::AbstractMatrix
	Z::AbstractMatrix
	counter::Integer
	freq::Integer

	function RandomizedQR(dim::Integer,M::Integer,m::Integer;freq::Integer=50)
		return new(
			M,
			m,
			Matrix{Float64}(undef,dim,M), #History: R^(dim × M)
			Matrix{Float64}(undef,dim,m), #reduced basis: R^(dim × m)
			Matrix{Float64}(undef,dim,m),
			Matrix{Float64}(undef,dim,m),
			Matrix{Float64}(undef,M,m),
			0,
			freq
		)
	end
end

function sketch_update!(strategy::RandomizedStrategy)
	if strategy.counter==strategy.freq || strategy.counter==0
		strategy.Z = randn((strategy.M, strategy.m))
		strategy.Ω = strategy.solutions * strategy.Z
		strategy.counter+=1
	else
		z = randn(strategy.m)
		@. strategy.Ω += strategy.solutions[:,end] * (z') - strategy.old_Ω
		cycle_and_replace!(strategy.Z,z; col=false)
	end

	strategy.old_Ω .= strategy.solutions[:,1] * (strategy.Z[1,:]')
end

function orderReduction!(strategy::RandomizedQR)
	
	sketch_update!(strategy)

	#Check out QRUpdate.jl?
	F = qr(strategy.Ω)
	strategy.basis .= @view Matrix(F.Q)[:,1:strategy.m];
end

mutable struct RandomizedSVD <: RandomizedStrategy
	M::Integer
	m::Integer
	solutions::AbstractMatrix
	basis::AbstractMatrix
	Ω::AbstractMatrix
	B::AbstractMatrix
	Q::AbstractMatrix
	old_Ω::AbstractMatrix
	Z::AbstractMatrix
	counter::Integer
	freq::Integer

	function RandomizedSVD(dim::Integer,M::Integer,m::Integer; freq::Integer=50)
		return new(
			M,
			m,
			Matrix{Float64}(undef,dim,M), #History: R^(dim × M)
			Matrix{Float64}(undef,dim,m), #reduced basis: R^(dim × m)
			Matrix{Float64}(undef,dim,m),
			Matrix{Float64}(undef,m,M),
			Matrix{Float64}(undef,dim,m),
			Matrix{Float64}(undef,dim,m),
			Matrix{Float64}(undef,M,m),
			0,
			freq
		)
	end
end

function orderReduction!(strategy::RandomizedSVD)
	#QR
	sketch_update!(strategy)

	F = qr(strategy.Ω)
	strategy.Q .= Matrix(F.Q)
	#svd
	mul!(strategy.B,strategy.Q',strategy.solutions)
	Fb = svd(strategy.B)
	
	strategy.basis .= @view (strategy.Q * Fb.U)[:,1:strategy.m]
end
