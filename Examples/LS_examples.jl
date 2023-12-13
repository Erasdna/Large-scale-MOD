using LinearAlgebra, Revise, Plots, InvertedIndices, LaTeXStrings, BenchmarkTools, JLD2, Random
include("../src/LSMOD.jl")
using .LSMOD


Random.seed!(parse(Int64,ARGS[3]))

const N=200
if ARGS[2]=="10e_3"
    const M=35
    const m=20
    const Nys_k = 14
    const Nys_p = 6
    Δt = 1e-3
elseif ARGS[2]=="10e_5"
    Δt = 1e-5
    const M=20
    const m=10
    const Nys_k = 7
    const Nys_p = 3
end

const t₀=0.1
const prob_size = 100

red = [m,2*m,5*m,10*m,100*m,1000*m]
prob = LSMOD.Example1.make_prob(prob_size)
if ARGS[1]=="Nystrom"
    orderReduction = LSMOD.Nystrom(prob.internal^2,M,Nys_k,Nys_p);
elseif ARGS[1]=="RQR"
    orderReduction=LSMOD.RandomizedQR(prob.internal^2,M,m);
elseif ARGS[1]=="RSVD"
    orderReduction=LSMOD.RandomizedSVD(prob.internal^2,M,m);
end
strat = LSMOD.UniformRowSampledLS
sols=Vector{Vector{Dict}}(undef,length(red))

dummy = LSMOD.solve(t₀, Δt , 40, deepcopy(prob), deepcopy(orderReduction));
base = LSMOD.solve(t₀, Δt , N, deepcopy(prob), deepcopy(orderReduction));

for (j,n) in enumerate(red)
    LS_strategy(A,rhs,args...) = strat(A,rhs,n,args...)
    LSMOD.solve(t₀, Δt , M+10, deepcopy(prob), deepcopy(orderReduction),LS_strategy);
    sols[j]=LSMOD.solve(t₀, Δt , N, deepcopy(prob), deepcopy(orderReduction),LS_strategy);
end


filename = pwd() * "/Examples/Data/LS/"*ARGS[2]*"_N="*string(prob_size)*"/LS_"*ARGS[1]*"_"*ARGS[3]*".jld2"

save(filename, 
    Dict("base" => base,
        "sols" => sols,
         "n" => red,
         "LS" => "Uniform",
         "m" => m,
         "M" => M,
         "dt" => Δt,
         "t0" => t₀,
         "N" => N,
         "prob_size" => prob_size
         )
    )
