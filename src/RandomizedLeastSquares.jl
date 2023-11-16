import StatsBase

"""
Row-sampling methods:
    - take max norm of each row
    - Use the function a directly:
        - Look at H¹ norm
        - Look at largest ∂a/∂t 

Should we sample with or without replacement?
"""

function FullLS(A::AbstractMatrix, rhs::AbstractVector,args...)
    return range(1,size(A,1)), copy(A), copy(rhs)
end

function UniformRowSampledLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer, args...)
    ind=StatsBase.sample(1:size(A,1),N;ordered=true)
    return ind, A[ind,:], rhs[ind]
end

function NormRowSampledLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer, args...)
    weights = StatsBase.weights(sum(A.^2;dims=2))
    ind=StatsBase.sample(1:size(A,1),weights,N;ordered=true)
    return ind, A[ind,:], rhs[ind]
end

function H1RowSampledLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer, problem::EllipticPDE, time::Number, args...)
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
    func(x::Tuple) = problem.a.∂a∂t(x,time)
    inner_grid = vcat(problem.inner_grid)

    #We take square of ∂a/∂t to pick the indices with the largest time derivative
    weights = StatsBase.weights(map(func,inner_grid).^2)
    ind=StatsBase.sample(1:size(A,1),weights,N;ordered=true)
    return ind, A[ind,:], rhs[ind]
end

function GaussianSketchLS(A::AbstractMatrix, rhs::AbstractVector, N::Integer,args...)
    Ω = randn(N,size(A,1))
    return N, Ω * A, Ω*rhs
end

export FullLS,UniformRowSampledLS,NormRowSampledLS,H1RowSampledLS,dtRowSampledLS,GaussianSketchLS