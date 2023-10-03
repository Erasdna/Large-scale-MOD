using LinearAlgebra, InvertedIndices, ForwardDiff
include("../src/LSMOD.jl")
import .LSMOD
using Plots;
using CurveFit, Printf

f(x::Vector) = sin(x[1])*sin(x[2])
f(x::Tuple) = f([x...])
a(x::Vector) = x[1]*x[2]
a(x::Tuple) = a([x...])

edge(n) = sort(collect([1:n; (n+1):n:(n^2 - 2*n + 1); (2*n):n:(n^2 -n); (n^2 - n + 1):n^2]))

N = [20,40,80,160]

#Testing only Δf(x,y):

exact1(x) = tr(ForwardDiff.hessian(f,[x...]))
Laplacian_error = Array{Float64}(undef,length(N))
real_N = copy(N)
for (i,n) ∈ enumerate(N)
    #We use the problem constructor to create the operators
    problem = LSMOD.EllipticPDE(n,0.0,1.0,0.0,1.0,nothing,nothing)
    inner = vcat(problem.grid)[InvertedIndices.Not(problem.edge)]

    f_val = map(f,inner)
    Δapprox = (problem.∂D.∂²x + problem.∂D.∂²y)*f_val
    Δexact = map(exact1,inner)
    ed1 = edge(n)
    ed2 = edge(n-2)
    real_N[i] = n^2 - length(ed1) - length(ed2)
    Δapprox = (Δapprox[Not(ed1)])[Not(ed2)]
    Δexact = (Δexact[Not(ed1)])[Not(ed2)]
    Laplacian_error[i] = norm(Δexact - Δapprox)
end

fit = curve_fit(LinearFit,log.(1 ./((N.+2).^2)),log.(Laplacian_error./real_N))
p1 = Plots.plot(1 ./ ( (N.+2).^2 ) ,Laplacian_error./real_N, 
    xaxis=:log,
    yaxis=:log, 
    title = "Δf convergence", 
    label = @sprintf("Convergence rate %1.2f",fit.coefs[2]),
    grid=true,
    lw=2)
Plots.plot(p1)
Plots.xlabel!("Grid size h")
Plots.ylabel!("(1/N)||Δf(x,y) - Δₙf(x,y)||₂")
Plots.savefig("Figures/Laplacian.png")

#Testing ∇a(x,y)∇f(x,y)
exact2(x) = ForwardDiff.gradient(f,[x...])' * ForwardDiff.gradient(a,[x...])
gradient_error = Array{Float64}(undef,length(N))
for (i,n) ∈ enumerate(N)
    problem = LSMOD.EllipticPDE(n,0.0,1.0,0.0,1.0,nothing,nothing)
    inner = vcat(problem.grid)[InvertedIndices.Not(problem.edge)]

    f_val = map(f,inner)
    ∇a(x::Tuple) = ForwardDiff.gradient(a,[x...])
    val = map(∇a,inner)

    ∂a∂x = getindex.(val,1)
    ∂a∂y = getindex.(val,2)

    Δapprox = (∂a∂x .* problem.∂D.∂x + ∂a∂x .* problem.∂D.∂y)*f_val
    Δexact = map(exact2,inner)

    ed1 = edge(n)
    ed2 = edge(n-2)
    Δapprox = (Δapprox[Not(ed1)])[Not(ed2)]
    Δexact = (Δexact[Not(ed1)])[Not(ed2)]
    gradient_error[i] = norm(Δexact - Δapprox)
end

fit2 = curve_fit(LinearFit,log.(1 ./ ((N.+2).^2)),log.(gradient_error./real_N))
p2=Plots.plot(1 ./ (N.+2).^2,gradient_error./real_N, 
    xaxis=:log,
    yaxis=:log, 
    title = "∇a∇f convergence", 
    label = @sprintf("Convergence rate %1.2f",fit2.coefs[2]),
    grid=true,
    lw=2)
Plots.plot(p2)
Plots.xlabel!("Grid size h")
Plots.ylabel!("(1/N)||∇a(x,y)∇f(x,y) - ∇a(x,y)∇ₙf(x,y)||₂")
Plots.savefig("Figures/grad_grad.png")

