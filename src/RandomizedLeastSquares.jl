import StatsBase

function FullLS(A::AbstractMatrix, rhs::AbstractVector,args...)
    """
        Baseline method, returns copy of system
    """
    return range(1,size(A,1)), copy(A), copy(rhs)
end

function UniformRowSampledLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer, args...)
    """
        Sample N rows uniformly
    """
    ind=StatsBase.sample(1:size(A,1),N;ordered=true)
    return ind, A[ind,:], rhs[ind]
end

function NormRowSampledLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer, args...)
    """
        Sample rows weighted by the norm of each row
    """
    weights = StatsBase.weights(sum(A.^2;dims=2))
    ind=StatsBase.sample(1:size(A,1),weights,N;ordered=true)
    return ind, A[ind,:], rhs[ind]
end

function H1RowSampledLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer, problem::EllipticPDE, time::Number, args...)
    """
        Sample rows of A based on the H1 norm of the wave speed a 
            - Context: We consider a problem like ∇(a∇f)=g
    """
    func(x::Tuple) = problem.a.a(x,time)
    func∂x(x::Tuple) = problem.a.∂a∂x(x,time)
    func∂y(x::Tuple) = problem.a.∂a∂y(x,time)

    inner_grid = vcat(problem.inner_grid)
    #Calculate square H₁ norm
    weights = StatsBase.weights(map(func,inner_grid).^2 + map(func∂x,inner_grid).^2 + map(func∂y,inner_grid).^2)
    ind=StatsBase.sample(1:size(A,1),weights,N;ordered=true)
    return ind, A[ind,:], rhs[ind]
end

function dtRowSampledLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer, problem::EllipticPDE, time::Number, args...)
    """
        Sample rows of A based on the largest value of the time-derivative of a
            - Context: We consider a problem like ∇(a∇f)=g
    """
    func(x::Tuple) = problem.a.∂a∂t(x,time)
    inner_grid = vcat(problem.inner_grid)

    #We take square of ∂a/∂t to pick the indices with the largest time derivative
    weights = StatsBase.weights(map(func,inner_grid).^2)
    ind=StatsBase.sample(1:size(A,1),weights,N;ordered=true)
    return ind, A[ind,:], rhs[ind]
end

function GaussianSketchLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer,args...)
    """
        Implements a Gaussian sketch as an alternative to row sampling
    """
    Ω = randn(N,size(A,1))
    return N, Ω * A, Ω*rhs
end

export FullLS,UniformRowSampledLS,NormRowSampledLS,H1RowSampledLS,dtRowSampledLS,GaussianSketchLS