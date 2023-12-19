using LinearAlgebra, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools, JLD2,Random
include("../../src/LSMOD.jl")
using .LSMOD

"""
    Runs the method for various problem sizes with and without row sampling 
    
    Options:
        1) Method: Nystrom, RQR, RSVD 
        2) Timestep: 10e_3 or 10e_5
        3) Row sampling: LS if yes
        4) Seed
"""
Random.seed!(parse(Int,ARGS[4]))

const N=200
if ARGS[2]=="10e_3"
    const M=35
    const m=20
    const nys_p = 7
    Δt = 1e-3
elseif ARGS[2]=="10e_5"
    Δt = 1e-5
    const M=20
    const m=10
    const nys_p=3
end

const t₀=0.1
#const prob_size = 100

if ARGS[1]=="Nystrom"
    orderReduction(prob_size,m) = LSMOD.Nystrom(prob_size^2,M,m-nys_p,nys_p);
elseif ARGS[1]=="RQR"
    orderReduction(prob_size,m)=LSMOD.RandomizedQR(prob_size^2,M,m);
elseif ARGS[1]=="RSVD"
    orderReduction(prob_size,m)=LSMOD.RandomizedSVD(prob_size^2,M,m);
end

if ARGS[3] == "LS"
    strat = LSMOD.UniformRowSampledLS
else
    strat = nothing
end


sizes = [200,150,125,100,75]
sols=Array{Vector{Dict}}(undef,length(sizes))
base=Array{Vector{Dict}}(undef,length(sizes))


for p in range(1,length(sizes))
    prob = LSMOD.Example1.make_prob(sizes[p])
    reduction = orderReduction(sizes[p],m)

    if !isnothing(strat)
        n = trunc(Int,0.005*sizes[p]^2)
        println(n)
        LS(A,rhs,args...) = strat(A,rhs,n,args...)
        LSMOD.solve(t₀, Δt , M+10, deepcopy(prob));
        base[p],_,_ = LSMOD.solve(t₀, Δt , N, deepcopy(prob));
    else 
        LS=nothing
        LSMOD.solve(t₀, Δt , M+10, deepcopy(prob) );
        base[p],_,_ = LSMOD.solve(t₀, Δt , N, deepcopy(prob));
    end
   
    LSMOD.solve(t₀, Δt , M+10, deepcopy(prob), deepcopy(reduction),LS)
    sols[p]=LSMOD.solve(t₀, Δt , N, deepcopy(prob), deepcopy(reduction),LS);
end

filename = pwd() * "/Examples/Data/Scaling/"*ARGS[2]*"_"*ARGS[3]*"/scaling_"*ARGS[1]*"_"*ARGS[4]*".jld2"

save(filename, 
    Dict("base" => base,
        "sols" => sols,
         "m" => m,
         "M" => M,
         "dt" => Δt,
         "t0" => t₀,
         "N" => N,
         "sizes" => sizes
         )
    )
