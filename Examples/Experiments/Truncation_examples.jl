using LinearAlgebra, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools, JLD2, Random
include("../src/LSMOD.jl")
using .LSMOD

"""
    Test out truncating the basis obtained by Randomized QR or Randomized SVD
"""
Random.seed!(parse(Int,ARGS[3]))

const N=1000
if ARGS[2]=="10e_3"
    const M=35
    const m=20
    const pf = 10
    Δt = 1e-3
elseif ARGS[2]=="10e_5"
    Δt = 1e-5
    const M=20
    const m=10
    const pf = 5
end

const t₀=0.1
const prob_size = 100

prob = LSMOD.Example1.make_prob(prob_size)
if ARGS[1]=="Nystrom"
    orderReduction(m,p) = LSMOD.Nystrom(prob_size^2,M,m-p,p);
elseif ARGS[1]=="RQR"
    orderReduction(m,p)=LSMOD.RandomizedQR(prob_size^2,M,m;p=p);
elseif ARGS[1]=="RSVD"
    orderReduction(m,p)=LSMOD.RandomizedSVD(prob_size^2,M,m;p=p);
end


sols=Array{Vector{Dict}}(undef,pf)

dummy = LSMOD.solve(t₀, Δt , M+10, deepcopy(prob) );
base,_,_ = LSMOD.solve(t₀, Δt , N, deepcopy(prob));

for p in range(1,pf)
    reduction = orderReduction(m,p-1)
    LSMOD.solve(t₀, Δt , M+10, deepcopy(prob), deepcopy(reduction))
    sols[p]=LSMOD.solve(t₀, Δt , N, deepcopy(prob), deepcopy(reduction));
end

filename = pwd() * "/Examples/Data/Truncation/"*ARGS[2]*"_truncation_"*ARGS[1]*"_"*ARGS[3]*".jld2"

save(filename, 
    Dict("base" => base,
        "sols" => sols,
         "m" => m,
         "M" => M,
         "pf" => pf,
         "dt" => Δt,
         "t0" => t₀,
         "N" => N,
         "prob_size" => prob_size
         )
    )
