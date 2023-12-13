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
    ret = Array{Float64}(undef,size(sols,1),6,N-start)
    for i in range(1,size(sols,1))
        #extract timing
        _,_,guess,_,_,tot = extract_timing(sols[i],start)
        ret[i,1,:] = guess
        ret[i,2,:] = tot
        #extract GMRES iterations
        ret[i,3,:] = extract_iters(sols[i],start)
        AQ,LS,_,red = extract_reduced_timing(sols[i],start)
        ret[i,4,:] = AQ
        ret[i,5,:] = LS 
        ret[i,6,:] = red
    end

    return ret
end

function extract_base(base,start,N)
    ret = Array{Float64}(undef,3,N-start)
    #extract timing 
    _,_,guess,_,_,tot = extract_timing(base,start)

    ret[1,:] = guess
    ret[2,:] = tot
    #extract GMRES iterations
    ret[3,:] = extract_iters(base,start)

    return ret
end

function Total_speedup_plot(fig,fig_guess,baseline,sols,sizes, tag)
    raw = Array{Float64}(undef,size(sols,1),size(sols,2))
    μ = Array{Float64}(undef,size(sols,2))
    σ = Array{Float64}(undef,size(sols,2))
    g_raw = Array{Float64}(undef,size(sols,1),size(sols,2))
    g_μ = Array{Float64}(undef,size(sols,2))
    g_σ = Array{Float64}(undef,size(sols,2))
    μ_GMRES = Array{Float64}(undef,size(sols,2))
    σ_GMRES = Array{Float64}(undef,size(sols,2))
    for i in range(1,size(sols,2))
        μ_GMRES[i] = mean(sum(sols[:,i,3,:],dims=2) ./ sum(baseline[:,3,:],dims=2))
        σ_GMRES[i] = std(sum(sols[:,i,3,:],dims=2) ./ sum(baseline[:,3,:],dims=2) )
    end
    println(tag)
    println(sizes)
    println("GMRES: ", μ_GMRES)
    println("σ: ", σ_GMRES)

    for i in range(1,size(sols,2))
        raw[:,i] = sum(baseline[:,2,:],dims=2) ./ sum(sols[:,i,2,:],dims=2)
        μ[i] = mean(raw[:,i])
        σ[i] = std(raw[:,i])
        g_raw[:,i] = sum(baseline[:,1,:],dims=2) ./ sum(sols[:,i,1,:],dims=2)
        g_μ[i] = mean(g_raw[:,i])
        g_σ[i] = std(g_raw[:,i])

        print("AQ: ", mean(sols[:,i,4,:])*1000, "±", std(sols[:,i,4,:])*1000)
        print(" LS: ", mean(sols[:,i,5,:])*1000, "±", std(sols[:,i,5,:])*1000)
        print(" red: ", mean(sols[:,i,6,:])*1000, "±", std(sols[:,i,6,:])*1000, "\n")
    end

    guess = mean(sum(sols[:,:,1,:],dims=3)./ sum(sols[:,:,2,:],dims=3),dims=1)
    guess_σ = std(sum(sols[:,:,1,:],dims=3)./ sum(sols[:,:,2,:],dims=3),dims=1)
    println("Guess: ", guess)
    println("σ: ", guess_σ)

    scatter!(fig,sizes, μ, yerror=σ, markersize=7,label=tag,markerstrokecolor=:auto)
    scatter!(fig_guess,sizes, g_μ, yerror=g_σ, markersize=7,label=tag,markerstrokecolor=:auto)
    return fig
end

function Guess_generation_stats(baseline,sols)
    raw = Array{Float64}(undef,size(sols,1),size(sols,2),3,size(sols,4))
    GMRES = Array{Float64}(undef,size(sols,1),size(sols,2))
    for i in range(1,size(sols,1))
        raw[i,:,:,:] = sols[i,:,4:6,:]*1000 # result in ms
        GMRES[i,:] = sum(sols[i,:,3,:],dims=2) ./ sum(baseline[:,3,:],dims=2)
    end
    μ = mean(raw,dims=4)[1,3,:,1]
    return μ,GMRES[1,3]
end


timestep="10e_5"
prob_size="100"
savefile = pwd() * "/Figures/Examples/LS/"*timestep*"_LS_comp_seeds"
#####
tt = pwd() * "/Examples/Data/LS/"*timestep*"_N="*prob_size*"/LS_Nystrom_33.jld2"
dat_tt = load(tt)
M = dat_tt["M"]
N = dat_tt["N"]
n = dat_tt["n"]
#####

start = 33
stop = 40
diff = stop - start + 1 
full_NYS = Array{Float64}(undef,diff,6,6,N-M)
full_QR = Array{Float64}(undef,diff,6,6,N-M)
full_SVD = Array{Float64}(undef,diff,6,6,N-M)
base_QR = Array{Float64}(undef,diff,3,N-M)
base_NYS = Array{Float64}(undef,diff,3,N-M)
base_SVD = Array{Float64}(undef,diff,3,N-M)

for i in range(start,stop)
    filename_NYS = pwd() * "/Examples/Data/LS/"*timestep*"_N="*prob_size*"/LS_Nystrom_"*string(i)*".jld2"
    filename_QR = pwd() * "/Examples/Data/LS/"*timestep*"_N="*prob_size*"/LS_RQR_"*string(i)*".jld2"
    filename_SVD = pwd() * "/Examples/Data/LS/"*timestep*"_N="*prob_size*"/LS_RSVD_"*string(i)*".jld2"
    
    dat_NYS = load(filename_NYS)
    dat_QR = load(filename_QR)
    dat_SVD = load(filename_SVD)

    full_NYS[i-32,:,:,:]= extract(dat_NYS["sols"],M+1,N+1)
    full_QR[i-32,:,:,:] = extract(dat_QR["sols"],M+1,N+1)
    full_SVD[i-32,:,:,:] = extract(dat_SVD["sols"],M+1,N+1)
    base_NYS[i-32,:,:] = extract_base(dat_NYS["base"],M+1,N+1)
    base_QR[i-32,:,:] = extract_base(dat_QR["base"],M+1,N+1)
    base_SVD[i-32,:,:] = extract_base(dat_SVD["base"],M+1,N+1)
end

#Speedup as a function of the problem size
fig = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Sampled rows",
        ylabel="Total Speedup",
        xaxis=:log,
    )

fig_guess = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Sampled rows",
        ylabel="Guess Speedup",
        xaxis=:log,
    )
Total_speedup_plot(fig,fig_guess, base_NYS,full_NYS,n,"Nyström")
Total_speedup_plot(fig,fig_guess, base_QR,full_QR,n, "Range Finder")
Total_speedup_plot(fig,fig_guess, base_SVD,full_SVD,n, "Randomized SVD")

savefig(fig,savefile*".png")
savefig(fig_guess,savefile*"_guess.png")