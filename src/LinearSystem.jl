using Base.Iterators, SparseArrays

export updateLinearSystem

function updateLinearSystem(problem::EllipticPDE, time::Float64)
    """
        Update linear system to be solved
    """
    func(x) = problem.a(x,time)
    a=flatmap(func,problem.grid)
    ∂a∂x = problem.∂D.∂x * collect(a)
    ∂a∂y = problem.∂D.∂y * collect(a)

    A = collect(a)' .* (problem.∂D.∂²x + problem.∂D.∂²y) +  (∂a∂x .* problem.∂D.∂x + ∂a∂y .* problem.∂D.∂y)
    dim = size(problem.grid)
    N = dim[1]
    edge = sort(collect([1:N; (N+1):N:(N^2 - 2*N + 1); (2*N):N:(N^2 -N); (N^2 - N + 1):N^2]))
    A[edge,:].=0
    A[edge,edge] = sparse(I,length(edge),length(edge))
    rhs_func(x) = problem.rhs(x,time)
    rhs = collect(flatmap(rhs_func,problem.grid))
    rhs[edge].=0.0

    return A, rhs
end