module Example1

using ..LSMOD
import ForwardDiff, LinearAlgebra

"""
    EXAMPLE PROBLEM 1:

    We consider the following problem:
        ∇⋅(a(x,y,t)∇f(x,y,t)) = rhs(x,y,t) ∀(x,y) ∈ Ω
        f(x,y,t)=0 ∀(x,y) ∈ ∂Ω
    With Ω ⊂ [0,1]² using 100 grid points in each direction
"""
struct waveSpeed 
    a::Function
    ∂a∂x::Function
    ∂a∂y::Function
    ∂a∂t::Function
end

#Wave speed
a(x::Vector, t) = exp(-(x[1] - 0.5)^2 - (x[2] - 0.5)^2) * cos(x[1] * t) + 2.1
a(x::Tuple, t) = a([x...],t)

∂a∂x(x::Vector, t) = (-1)*exp(-(x[1] - 0.5)^2 - (x[2] - 0.5)^2)*(t*sin(x[1] * t) + 2*(x[1] - 0.5)*cos(x[1] * t))
∂a∂x(x::Tuple, t) = ∂a∂x([x...],t)

∂a∂y(x::Vector, t) = (-1)*exp(-(x[1] - 0.5)^2 - (x[2] - 0.5)^2)*2*(x[2] - 0.5)* cos(x[1] * t)
∂a∂y(x::Tuple, t) = ∂a∂y([x...],t)

∂a∂t(x::Vector, t) = exp(-(x[1] - 0.5)^2 - (x[2] - 0.5)^2) * (-1*x[1])*sin(x[1] * t)
∂a∂t(x::Tuple, t) = ∂a∂t([x...],t)

a1 = waveSpeed(a,∂a∂x,∂a∂y,∂a∂t)

#Exact solution
exact1(x::Vector, t) = sin(4*pi*x[1])*sin(4*pi*x[2])*(1 + sin(15*pi*x[1]*t)*sin(3*pi*x[2]*t)*exp(-(x[1]-0.5)^2 - (x[2]-0.5)^2 - 0.25^2)) 
exact1(x::Tuple, t) = exact1([x...],t) 

#We calculate the rhs function using automatic differentiation
function rhs1(x::Tuple,t,a,exact)
    frozen_a(y::Vector) = a(y,t)
    frozen_exact(y::Vector) = exact(y,t)
    
    x_vec = [x...]
    d1 = ForwardDiff.gradient(frozen_a,x_vec)' * ForwardDiff.gradient(frozen_exact,x_vec)
    d2 = frozen_a(x_vec) * LinearAlgebra.tr(ForwardDiff.hessian(frozen_exact,x_vec))
    return d1 + d2
end

g1(x::Tuple,t) = rhs1(x,t,a,exact1)

function make_prob(N::Integer)
    return LSMOD.EllipticPDE(
                N, # discretisation (each direction)
                0.0, # xmin
                1.0, # xmax
                0.0, # ymin
                1.0, # ymax
                a1, # wave speed (~ish)
                g1, # rhs
            )
end

end
