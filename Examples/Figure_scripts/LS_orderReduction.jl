using JLD2,Plots, LaTeXStrings, Statistics
include(pwd()*"/src/LSMOD.jl")
using .LSMOD

"""
Quantities of interest:
    1) Total speedup
    2) Total (average) GMRES iteration speedup
    3) AQ,LS speedup  
"""
function extract_timing(data,start)
    precon = [data[el][:timing][:preconditioner] for el in range(start,length(data))]
    basis = [haskey(data[el][:timing],:basis) ? data[el][:timing][:basis] : 0.0 for el in range(start,length(data))]
    guess = [haskey(data[el][:timing],:guess) ? data[el][:timing][:guess] : 0.0 for el in range(start,length(data))]
    gmres = [data[el][:timing][:gmres] for el in range(start,length(data))]
    non_gmres = basis + guess
    tot = basis + guess + gmres 
    return precon,basis,guess,gmres, non_gmres,tot
end

extract_iters(v,start) = [v[el][:history].niter for el in range(start,length(v))]

function extract_reduced_timing(data,start)
    red = [data[el][:timing][:guess_detailed][:red] for el in range(start,length(data))]
    AQ = [data[el][:timing][:guess_detailed][:AQ] for el in range(start,length(data))]
    LS = [data[el][:timing][:guess_detailed][:LS] for el in range(start,length(data))]
    IG = [data[el][:timing][:guess_detailed][:IG] for el in range(start,length(data))]
    return AQ,LS,IG,red
end

function extract(sols,start,N)
    ret = Array{Float64}(undef,size(sols,1),size(sols,2),6,N-start)
    for i in range(1,size(sols,1))
        for j in range(1,size(sols,2))
            #extract timing 
            _,_,guess,_,_,tot = extract_timing(sols[i,j],start)
            ret[i,j,1,:] = guess
            ret[i,j,2,:] = tot
            #extract GMRES iterations
            ret[i,j,3,:] = extract_iters(sols[i,j],start)
            #extract guess breakdown
            AQ,LS,_,red = extract_reduced_timing(sols[i,j],start)
            ret[i,j,4,:] = AQ 
            ret[i,j,5,:] = LS 
            ret[i,j,6,:] = red
        end
    end

    return ret
end

function extract_base(base,start,N)
    ret = Array{Float64}(undef,1,6,N-start)
    #extract timing 
    _,_,guess,_,_,tot = extract_timing(base,start)
    ret[1,1,:] = guess
    ret[1,2,:] = tot
    #extract GMRES iterations
    ret[1,3,:] = extract_iters(base,start)
    #extract guess breakdown
    AQ,LS,_,red = extract_reduced_timing(base,start)
    ret[1,4,:] = AQ 
    ret[1,5,:] = LS 
    ret[1,6,:] = red

    return ret
end


function Total_speedup_plot(baselines,sols,red,tags)
    raw_NYS = Array{Float64}(undef,size(sols[1],1),size(sols[1],2),2,size(sols[1],4))
    raw_QR = Array{Float64}(undef,size(sols[1],1),size(sols[1],2),2,size(sols[1],4))
    raw_SVD = Array{Float64}(undef,size(sols[1],1),size(sols[1],2),2,size(sols[1],4))

    for i in range(1,size(sols[3],1))
        raw_NYS[i,:,:,:] = baselines[1][:,1:2,:] ./ sols[1][i,:,1:2,:]
        raw_QR[i,:,:,:] = baselines[2][:,1:2,:] ./ sols[2][i,:,1:2,:]
        raw_SVD[i,:,:,:] = baselines[3][:,1:2,:] ./ sols[3][i,:,1:2,:]
    end
    μ_NYS = mean(raw_NYS,dims=4)
    μ_QR = mean(raw_QR,dims=4)
    μ_SVD = mean(raw_SVD,dims=4)
    μs = [μ_NYS,μ_QR,μ_SVD]

    fig1 = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Rows",
        ylabel="Guess Speedup",
        xaxis=:log
    )
    fig2 = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Rows",
        ylabel="Total Speedup",
        xaxis=:log,
    )
    for i in range(1,length(μs))
        scatter!(fig1,red, μs[i][1,:,1],label = tags[i],markersize=5.0)
        scatter!(fig2,red, μs[i][1,:,2],label = tags[i],markersize=5.0)
    end

    return fig1,fig2
end

function Guess_generation_stats(baseline,sols)
    raw = Array{Float64}(undef,size(sols,1),size(sols,2),3,size(sols,4))
    GMRES = Array{Float64}(undef,size(sols,1),size(sols,2))
    for i in range(1,size(sols,1))
        raw[i,:,:,:] = sols[i,:,4:6,:]*1000 # result in ms
        GMRES[i,:] = sum(sols[i,:,3,:],dims=2) ./ sum(baseline[:,3,:],dims=2)
    end
    μ = mean(raw,dims=4)[1,2,:,1]
    return μ,GMRES[1,2]
end

filename_NYS = pwd() * "/Examples/Data/LS/"*ARGS[1]*"_LS_Nystrom.jld2"
filename_QR = pwd() * "/Examples/Data/LS/"*ARGS[1]*"_LS_RQR.jld2"
filename_SVD = pwd() * "/Examples/Data/LS/"*ARGS[1]*"_LS_RSVD.jld2"
savefile = pwd() * "/Figures/Examples/LS/"*ARGS[1]*"_LS_comp"

dat_NYS = load(filename_NYS)
dat_QR = load(filename_QR)
dat_SVD = load(filename_SVD)
M = dat_NYS["M"]
N = dat_NYS["M"]
n = dat_NYS["n"]
tags = dat_NYS["LS"]

ind = [1,2,3,4]

mat_NYS = extract(dat_NYS["sols"],M+1,501)
mat_QR = extract(dat_QR["sols"],M+1,501)
mat_SVD = extract(dat_SVD["sols"],M+1,501)


base_NYS = extract_base(dat_NYS["base"],M+1,501)
base_QR = extract_base(dat_QR["base"],M+1,501)
base_SVD = extract_base(dat_SVD["base"],M+1,501)

stats_NYS = Guess_generation_stats(base_NYS,mat_NYS)
println("Nystrom")
println(stats_NYS[1])
println(stats_NYS[2])
stats_QR = Guess_generation_stats(base_QR,mat_QR)
println("QR")
println(stats_QR[1])
println(stats_QR[2])
stats_SVD = Guess_generation_stats(base_SVD,mat_SVD)
println("SVD")
println(stats_SVD[1])
println(stats_SVD[2])

fig_Guess,fig_Total = Total_speedup_plot([base_NYS,base_QR,base_SVD], [mat_NYS,mat_QR,mat_SVD],n,["Nystrom", "Range Finder", "Randomized SVD"])
savefig(fig_Guess,savefile*"_Guess.png")
savefig(fig_Total,savefile*"_Total.png")