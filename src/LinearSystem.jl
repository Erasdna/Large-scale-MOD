using Base.Iterators

export updateLinearSystem

function updateLinearSystem(problem::EllipticPDE, time::Float64)
    """
        Update linear system to be solved
    """
    func(x) = problem.a(x,time)
    a=flatmap(func,problem.grid)
    ∂a∂x = problem.∂D.∂x * collect(a)
    ∂a∂y = problem.∂D.∂y * collect(a)

    A = collect(a)' .* (problem.∂D.∂²x + problem.∂D.∂²y) #+  (∂a∂x .* problem.∂D.∂x + ∂a∂y .* problem.∂D.∂y)
    rhs_func(x) = problem.rhs(x,time)
    rhs = collect(flatmap(rhs_func,problem.grid))

    return A, rhs
end