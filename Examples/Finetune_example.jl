using LinearAlgebra, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools, JLD2, Random
include("../src/LSMOD.jl")
using .LSMOD

Random.seed!(parse(Int64,ARGS[5]))

const N=200
if ARGS[2]=="10e_3"
    Δt = 1e-3
    Ms = [35,45,55]
elseif ARGS[2]=="10e_5"
    Δt = 1e-5
    Ms = [20,30,40]
end

const t₀=0.1
const prob_size = parse(Int64,ARGS[4])

nys_p(m) = trunc(Int,floor(m*1/3))

if ARGS[1]=="Nystrom"
    orderReduction(prob_size,m,M) = LSMOD.Nystrom(prob_size^2,M,m-nys_p(m),nys_p(m));
elseif ARGS[1]=="RQR"
    orderReduction(prob_size,m,M)=LSMOD.RandomizedQR(prob_size^2,M,m);
elseif ARGS[1]=="RSVD"
    orderReduction(prob_size,m,M)=LSMOD.RandomizedSVD(prob_size^2,M,m);
end

prob = LSMOD.Example1.make_prob(prob_size)
n = trunc(Int,0.005*prob_size^2)

if ARGS[3] == "LS"
    LS(A,rhs,args...) = LSMOD.UniformRowSampledLS(A,rhs,n,args...)
else
    LS = nothing
end

m_frac = [0.3,0.5,0.7,0.9]
sols=Array{Vector{Dict}}(undef,length(Ms),length(m_frac))

LSMOD.solve(t₀, Δt , Ms[end], deepcopy(prob));
base,_,_ = LSMOD.solve(t₀, Δt , N, deepcopy(prob));

for (i,M) in enumerate(Ms)
    println("M: ",M)
    for (j,frac) in enumerate(m_frac) 
        m = trunc(Int,frac*M)
        println("m: ", m, " p: ", nys_p(m))
        reduction = orderReduction(prob_size,m,M)
    
        LSMOD.solve(t₀, Δt , M+10, deepcopy(prob), deepcopy(reduction),LS)
        sols[i,j]=LSMOD.solve(t₀, Δt , N, deepcopy(prob), deepcopy(reduction),LS);
    end
end

filename = pwd() * "/Examples/Data/Finetune/"*ARGS[2]*"_N="*ARGS[4]*"/finetune_"*ARGS[1]*"_"*ARGS[3]*"_"*ARGS[5]*".jld2"

save(filename, 
    Dict("base" => base,
        "sols" => sols,
         "m" => m_frac,
         "M" => Ms,
         "dt" => Δt,
         "t0" => t₀,
         "N" => N
        )
    )
