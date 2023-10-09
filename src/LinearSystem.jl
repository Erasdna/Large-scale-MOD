import InvertedIndices

export updateLinearSystem

function updateLinearSystem(problem::EllipticPDE, time::Float64)
    """
        Update linear system to be solved
    """
    func(x::Vector) = problem.a(x,time)
    func(x::Tuple) = func([x...])

    inner_grid = vcat(problem.grid)[InvertedIndices.Not(problem.edge)]
    a=map(func,inner_grid)
    ∇a(x::Tuple) = ForwardDiff.gradient(func,[x...])
    val = map(∇a,inner_grid)

    ∂a∂x = getindex.(val,1)
    ∂a∂y = getindex.(val,2)

    A = a' .* (problem.∂D.∂²x + problem.∂D.∂²y) +  (∂a∂x .* problem.∂D.∂x + ∂a∂y .* problem.∂D.∂y)
    rhs_func(x) = problem.rhs(x,time)
    rhs = map(rhs_func,inner_grid)

    return A, rhs
end