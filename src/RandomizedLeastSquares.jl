import StatsBase

#Row sampling strategy
function Lstsq_row(A::AbstractMatrix,rhs::AbstractVector,N::Integer)
    RHS = copy(rhs)
    ind=StatsBase.sample(1:size(A)[1],N;ordered=true)
    mat = @view A[ind,:]
    _,ret,_ =LAPACK.gels!('N',A,RHS)
    return ret 
end
#Column sampling strategy
