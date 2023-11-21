using JLD2,Plots, LaTeXStrings
include(pwd()*"/src/LSMOD.jl")
using .LSMOD

filename = pwd() * "/Examples/Data/10e_3_update.jld2"
savefile = pwd() * "/Figures/Examples/Projection/10e_3_update"
dat = load(filename)

start = dat["M"]+1
extract_proj(v,start) = [v[el][:proj] for el in range(start,length(v))]
extract_proj_X(v,start) = [v[el][:proj_X] for el in range(start,length(v))]

#base = extract_proj_X(dat["POD"],start)
Nystrom = extract_proj(dat["Nystrom"],start)
POD = extract_proj(dat["POD"],start)
RangeFinder = extract_proj(dat["RandQR"],start)
RandomizedSVD = extract_proj(dat["RandSVD"],start)

step=10
ind = 1:step:(dat["N"]-start)
title = L"\Delta t = "*string(dat["dt"])*", M="*string(dat["M"])*", m="*string(dat["m"])

fig = scatter(ind, 
            [Nystrom[ind],POD[ind],RangeFinder[ind],RandomizedSVD[ind]], 
            label= ["Nystrom" "POD" "Range Finder" "Randomized SVD"], 
            lw=2,
            grid=true,
            yaxis=:log10,
            guidefontsize=14,
            tickfontsize=12,
            legendfontsize=12,
            legend=:bottomright,
            xlabel="Timestep",
            ylabel=L"||X - X_m||_F/||X||_F")
Plots.savefig(fig,savefile*".png")
Plots.savefig(fig,savefile*".svg")