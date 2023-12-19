using JLD2,Plots, LaTeXStrings
include(pwd()*"/src/LSMOD.jl")
using .LSMOD

filename = pwd() * "/Examples/Data/10e_3/final_33_2.jld2"
savefile = pwd() * "/Figures/Examples/GMRES/10e_3_final_33_2"
dat = load(filename)

start = dat["M"]+1
extract_iters(v,start) = [v[el][:history].niter for el in range(start,length(v))]

base = extract_iters(dat["base"],start)
Nystrom = extract_iters(dat["Nystrom"],start)
POD = extract_iters(dat["POD"],start)
RangeFinder = extract_iters(dat["RandQR"],start)
RandomizedSVD = extract_iters(dat["RandSVD"],start)

step=10
ind = 1:step:(dat["N"]-start)
title = L"\Delta t = "*string(dat["dt"])*", M="*string(dat["M"])*", m="*string(dat["m"])

fig = scatter(ind, 
            [Nystrom[ind],POD[ind],RangeFinder[ind],RandomizedSVD[ind],base[ind]], 
            label= ["Nystrom" "POD" "Range Finder" "Randomized SVD" "Base"], 
            lw=2,
            grid=true,
            guidefontsize=14,
            tickfontsize=12,
            legendfontsize=12,
            markerstrokecolor=:auto,
            xlabel="Iteration",
            ylabel="GMRES iterations")
Plots.savefig(fig,savefile*".png")
