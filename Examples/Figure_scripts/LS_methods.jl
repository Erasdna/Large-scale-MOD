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

function GMRES_plot(baseline,sols,red,tags)
    raw = Array{Float64}(undef,size(sols,1),size(sols,2),size(sols,4))
    for i in range(1,size(sols,1))
        raw[i,:,:] = baseline[:,3,:] ./ sols[i,:,3,:]
    end
    μ = mean(raw,dims=3)
    σ = std(raw,dims=3)
    fig = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Rows",
        ylabel="GMRES Speedup",
        xaxis=:log
    )
    for i in range(1,size(raw,1))
        scatter!(fig,red, μ[i,:],label = tags[i],yerror=σ[i,:])
    end

    return fig
end

function Total_speedup_plot(baseline,sols,red,tags)
    raw = Array{Float64}(undef,size(sols,1),size(sols,2),2,size(sols,4))
    for i in range(1,size(sols,1))
        raw[i,:,:,:] = baseline[:,1:2,:] ./ sols[i,:,1:2,:]
    end
    μ = mean(raw,dims=4)
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
        xaxis=:log
    )
    for i in range(1,size(raw,1))
        scatter!(fig1,red, μ[i,:,1],label = tags[i])
        scatter!(fig2,red, μ[i,:,2],label = tags[i])
    end

    return fig1,fig2
end

function Guess_generation_stats(baseline,sols)
    raw = Array{Float64}(undef,size(sols,1),size(sols,2),2,size(sols,4))
    for i in range(1,size(sols,1))
        raw[i,:,:,:] = baseline[:,4:5,:] ./ sols[i,:,4:5,:]
    end
    μ = mean(raw,dims=4)
    σ = std(raw,dims=4)
    return μ,σ
end

filename = pwd() * "/Examples/Data/LS/10e_3_LS_Nystrom_200.jld2"
savefile = pwd() * "/Figures/Examples/LS/10e_3_LS_Nystrom_200"
dat = load(filename)
M = dat["M"]
N = dat["M"]
n = dat["n"]
tags = dat["LS"]

ind = [1,2]

mat = extract(dat["sols"],M+1,501)
base = extract_base(dat["base"],M+1,501)
fig = GMRES_plot(base, mat[ind,:,:,:],n,tags[ind])
fig2,fig3 = Total_speedup_plot(base, mat[ind,:,:,:],n,tags[ind])