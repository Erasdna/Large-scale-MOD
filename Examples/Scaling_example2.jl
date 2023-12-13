using LinearAlgebra, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools, JLD2
include("../src/LSMOD.jl")
using .LSMOD

const N=200
if ARGS[2]=="10e_3"
    const M=35
    const m=20
    Δt = 1e-3
elseif ARGS[2]=="10e_5"
    Δt = 1e-5
    const M=20
    const m=10
end

const t₀=0.1
const prob_size = 100

nys_p(m) = trunc(Int,ceil(m*1/3))
if ARGS[1]=="Nystrom"
    orderReduction(prob_size,m,M) = LSMOD.Nystrom(prob_size^2,M,m-nys_p(m),nys_p(m));
elseif ARGS[1]=="RQR"
    orderReduction(prob_size,m,M)=LSMOD.RandomizedQR(prob_size^2,M,m);
elseif ARGS[1]=="RSVD"
    orderReduction(prob_size,m,M)=LSMOD.RandomizedSVD(prob_size^2,M,m);
end

if ARGS[3] == "LS"
    strat = LSMOD.UniformRowSampledLS
else
    strat = nothing
end


sizes = [200,160,120,80]
sols=Array{Vector{Dict}}(undef,length(sizes))
base=Array{Vector{Dict}}(undef,length(sizes))


for p in range(1,length(sizes))
    prob = LSMOD.Example1.make_prob(sizes[p])
    scale_m = trunc(Int,m*(sizes[p]/100)^2)
    scale_M = trunc(Int,M*(sizes[p]/100)^2)
    reduction = orderReduction(sizes[p],scale_m,scale_M)

    if !isnothing(strat)
        n = trunc(Int,0.005*sizes[p]^2)
        println(n)
        LS(A,rhs,args...) = strat(A,rhs,n,args...)
        LSMOD.solve(t₀, Δt , M+10, deepcopy(prob),deepcopy(reduction) );
        base[p] = LSMOD.solve(t₀, Δt , N, deepcopy(prob),deepcopy(reduction));
    else 
        LS=nothing
        LSMOD.solve(t₀, Δt , scale_M+10, deepcopy(prob) );
        base[p],_,_ = LSMOD.solve(t₀, Δt , N + scale_M, deepcopy(prob));
    end
   
    LSMOD.solve(t₀, Δt , scale_M+10, deepcopy(prob), deepcopy(reduction),LS)
    sols[p]=LSMOD.solve(t₀, Δt , N + scale_M, deepcopy(prob), deepcopy(reduction),LS);
end

filename = pwd() * "/Examples/Data/Scaling/"*ARGS[2]*"_scaling_"*ARGS[1]*"_"*ARGS[3]*"_mM.jld2"

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
