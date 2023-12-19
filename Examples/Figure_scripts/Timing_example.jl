using LinearAlgebra,ForwardDiff, Revise, Plots, InvertedIndices
include("../src/LSMOD.jl")
using .LSMOD

function extract_timing(data,start)
    precon = [data[el][:timing][:preconditioner] for el in range(start,length(data))]
    basis = [haskey(data[el][:timing],:basis) ? data[el][:timing][:basis] : 0.0 for el in range(start,length(data))]
    guess = [haskey(data[el][:timing],:guess) ? data[el][:timing][:guess] : 0.0 for el in range(start,length(data))]
    gmres = [data[el][:timing][:gmres] for el in range(start,length(data))]
    non_gmres = basis + guess
    return precon,basis,guess,gmres, non_gmres
end

function extract_reduced_timing(data,start)
    AQ = [data[el][:timing][:guess_detailed][:AQ] for el in range(start,length(data))]
    LS = [data[el][:timing][:guess_detailed][:LS] for el in range(start,length(data))]
    IG = [data[el][:timing][:guess_detailed][:IG] for el in range(start,length(data))]
    return AQ,LS,IG
end

function solver_time(data,M::Integer, title::String)
    precon,basis,guess,gmres,non_gmres = extract_timing(data,M)

    #Proportion of each timing
    fig1 = scatter(M:M+length(precon), 
            [cumsum(precon),cumsum(basis),cumsum(guess),cumsum(gmres)], 
            title = title, 
            label= ["Preconditioner" "Basis" "Guess" "GMRES"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Cumulative time [s]")
    fig2 = scatter(M:M+length(precon), 
            [non_gmres,gmres], 
            title = title, 
            label= ["Initial guess" "GMRES"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Cumulative time [s]")
    return fig1,fig2
end

function method_comp(base,method1, tag1::String, method2, tag2::String, method3,tag3,method4,tag4,M)
    b = extract_timing(base,M)
    m1 = extract_timing(method1,M)
    m2 = extract_timing(method2,M)
    m3 = extract_timing(method3,M)
    m4 = extract_timing(method4,M)
    fig1 = scatter(M:M+length(b[1]), 
            [b[end] + b[end-1],m1[end] + m1[end-1],m2[end] + m2[end-1],m3[end] + m3[end-1],m4[end] + m4[end-1]], 
            label= ["base" tag1 tag2 tag3 tag4], 
            lw=2,
            xlabel="Timestep",
            ylabel="Time [s]")
    bb=cumsum(b[end] + b[end-1])
    fig2 = scatter(M:M+length(b[1]), 
            [bb./cumsum(m1[end] + m1[end-1]),bb./cumsum(m2[end] + m2[end-1]),bb./cumsum(m3[end] + m3[end-1]),bb./cumsum(m4[end] + m4[end-1])], 
            label= [tag1 tag2 tag3 tag4], 
            lw=2,
            xlabel="Timestep",
            ylabel="Speedup")
    return fig1,fig2
end

function reduced_time(data,M::Integer, title::String)
    AQ,LS,IG = extract_reduced_timing(data,M)

    #Proportion of each timing
    fig1 = scatter(M:M+length(AQ), 
            [AQ,LS,IG], 
            title = title, 
            label= ["AQ" "Least squares" "Qs"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Time [s]")
    return fig1
end

"""
    We reproduce Figure 1 in [cite paper]
    Settings:
        - t=2.3 second, 200 timesteps Δt = 10⁻³ and 10⁻⁵
        - At Δt = 10⁻³ we use M=20, m=10
        - At Δt = 10⁻⁵ we use M=35, m=20
        - For each timestep we supply a reference method using the 
          previous step as initial guess for GMRES
        - Other settings:
            - GMRES without restarting
            - Tolerance: ||Ax - b||₂/||b||₂ ≤ 10⁻⁷
            - Incomplete LU preconditioner with no fill in
            - Fourth order discretisation scheme in space
"""

N=200
steps = 1:200
M_3=35
m_3=20
t₀=0.0
Δt=1e-3

prob = LSMOD.Example1.prob
#Δt = 10⁻³
sol_base_3,_ = LSMOD.solve(t₀, Δt, N, prob)
RandNYS = LSMOD.Nystrom(prob.internal^2,M_3,14,7)
sol_RNYS_3 = LSMOD.solve(t₀, Δt_3 , N, prob, RandNYS)
pod = LSMOD.POD(prob.internal^2,M_3,m_3)
sol_POD_3 = LSMOD.solve(t₀, Δt_3 , N, prob, pod)
RQR=LSMOD.RandomizedQR(prob.internal^2,M_3,m_3)
sol_Rand_3 = LSMOD.solve(t₀, Δt_3 , N, prob, RQR)
RSVD=LSMOD.RandomizedSVD(prob.internal^2,M_3,m_3)
sol_RandSVD_3 = LSMOD.solve(t₀, Δt_3 , N, prob, RSVD)

start = 40 

function save_img(list,start,tag)
    f1,f2 = solver_time(list,start,tag)
    savefig(f1, "Figures/Timing/"*tag*"1.png")
    savefig(f2, "Figures/Timing/"*tag*"2.png")

    if tag!="base"
        f3 = reduced_time(list,start,tag)
        savefig(f3, "Figures/Timing/"*tag*"3.png")
    end
end

save_img(sol_base_3,start,"base")
save_img(sol_POD_3,start,"POD")
save_img(sol_Rand_3,start,"RandomizedQR")
save_img(sol_RandSVD_3,start,"RandomizedSVD")
save_img(sol_RNYS_3,start,"Nystrom")


f3,f4 = method_comp(sol_base_3,sol_POD_3,"POD",sol_Rand_3,"RandomizedQR", sol_RandSVD_3, "RandomizedSVD", sol_RNYS_3,"Nystrom",start)
savefig(f3, "Figures/Timing/comp.png")
savefig(f4, "Figures/Timing/Cumulative_comp.png")
