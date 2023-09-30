export updateLinearSystem

function updateLinearSystem(problem::EllipticPDE, time::Float64)
    """
        Update linear system to be solved
    """
    func(x::Vector) = problem.a(x,time)
    func(x::Tuple) = problem.a(x,time)

    a=flatmap(func,problem.grid)
    ∇a(x::Tuple) = ForwardDiff.gradient(func,[x...])
    val = collect(flatmap(∇a,problem.grid))

    ∂a∂x = val[1:2:end] #getfield.(val,1)
    ∂a∂y = val[2:2:end] #getfield.(val,2)

    A = collect(a)' .* (problem.∂D.∂²x + problem.∂D.∂²y) +  (∂a∂x .* problem.∂D.∂x + ∂a∂y .* problem.∂D.∂y)
    edge = problem.edge
    A[edge,:].=0
    A[edge,edge] = sparse(I,length(edge),length(edge))
    rhs_func(x) = problem.rhs(x,time)
    rhs = collect(flatmap(rhs_func,problem.grid))
    rhs[edge].=0.0

    return A, rhs
end