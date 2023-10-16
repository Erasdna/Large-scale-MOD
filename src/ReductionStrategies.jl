export POD!, RandomizedQR!

function POD!(basis,solutions, M, m)
	dims = size(solutions)
	@assert(dims[2] == M)

	#mat = @view solutions[:, end-M+1:end]

	F = svd(solutions)
	basis .= (F.U)[:, 1:m]
end

function RandomizedQR!(basis,solutions, M, m)
	dims = size(solutions)
	@assert(dims[2] == M)

	mat = @view solutions[:, end-M+1:end]

	Z = randn((M, m))
	Ω = mat * Z
	
	#Check out QRUpdate.jl?
	F = qr(Ω)
	basis .= @view Matrix(F.Q)[:,1:m];
end

