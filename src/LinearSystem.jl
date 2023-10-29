import InvertedIndices

export updateLinearSystem!

function updateLinearSystem!(problem::EllipticPDE, time::Float64)
    """
        Update linear system to be solved
    """
    func(x::Vector) = problem.a(x,time)
    func(x::Tuple) = func([x...])

    inner_grid = vcat(problem.inner_grid)
    problem.update.a_vec .= map(func,inner_grid)
    ∇a(x::Tuple) = ForwardDiff.gradient(func,[x...])
    val = map(∇a,inner_grid)

    problem.update.∂a∂x_vec .= getindex.(val,1)
    problem.update.∂a∂y_vec .= getindex.(val,2)

    #Supercharged update of the nonzero entries of A
    #We iterate over the nonzero elements of the hessian because
    #the sparsity structure of this matrix is constant
    @inbounds @simd for col in 1:size(problem.∂D.Δ,2)
        for r in nzrange(problem.∂D.Δ,col)
            row = rowvals(problem.∂D.Δ)[r]
            problem.update.A[row,col] = problem.update.a_vec[row]*nonzeros(problem.∂D.Δ)[r] + problem.update.∂a∂x_vec[col]*problem.∂D.∂x[row,col] + problem.update.∂a∂y_vec[col]*problem.∂D.∂y[row,col] 
        end
    end

    rhs_func(x) = problem.rhs(x,time)
    problem.update.rhs_vec .= map(rhs_func,inner_grid);
end

