import InvertedIndices

export updateLinearSystem!

function updateLinearSystem!(problem::EllipticPDE, time::Number)
    """
        Update linear system to be solved. We update quantities inside the problem struct 

        Input:
            - problem (EllipticPDE): struct containing information about problem to be solved
            - time (Number): Current time
        Output:
            Nothing
    """

    #"Freeze" functions in time to simplify map
    func(x::Tuple) = problem.a.a(x,time)
    func∂x(x::Tuple) = problem.a.∂a∂x(x,time)
    func∂y(x::Tuple) = problem.a.∂a∂y(x,time)

    inner_grid = vcat(problem.inner_grid)

    #Calculate function values at every point in the mesh
    problem.update.a_vec .= map(func,inner_grid)
    problem.update.∂a∂x_vec .= map(func∂x,inner_grid)
    problem.update.∂a∂y_vec .= map(func∂y,inner_grid)

    #Supercharged update of the nonzero entries of A
    #We iterate over the nonzero elements of the hessian because
    #the sparsity structure of this matrix is constant
    @inbounds @simd for col in 1:size(problem.∂D.Δ,2)
        for r in nzrange(problem.∂D.Δ,col)
            row = rowvals(problem.∂D.Δ)[r]
            problem.update.A[row,col] = problem.update.a_vec[row]*nonzeros(problem.∂D.Δ)[r] + problem.update.∂a∂x_vec[row]*problem.∂D.∂x[row,col] + problem.update.∂a∂y_vec[row]*problem.∂D.∂y[row,col] 
        end
    end

    #Update right hand side
    rhs_func(x) = problem.rhs(x,time)
    problem.update.rhs_vec .= map(rhs_func,inner_grid);
end

