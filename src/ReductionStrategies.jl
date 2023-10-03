export POD!, RandomizedQR!
import Statistics

# mutable struct ReducedModel
# 	M::Integer
# 	m::Integer
# 	basis::Any
# 	LinAlgObject::Any

# 	function ReducedModel(M::Integer, m::Integer)
# 		new(M, m, Nothing, Nothing)
# 	end
# end

function POD!(solutions, M, m)
	dims = size(solutions)
	@assert(dims[2] >= M)

	mat = solutions[:, end-M+1:end]
	F = svd(mat)
	return (F.U)[:, 1:m]
end

function RandomizedQR!(solutions, M, m)
	dims = size(solutions)
	@assert(dims[2] >= M)

	mat = solutions[:, end-M+1:end]
	Z = randn((M, m))
	Ω = mat * Z

	#Check out QRUpdate.jl?
	F = qr(Ω)
	return (F.Q)[:,1:m]
end

