using JLD2,Plots, LaTeXStrings, Statistics
include(pwd()*"/src/LSMOD.jl")
using .LSMOD

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


function extract(sols,start,N)
    ret = Array{Float64}(undef,size(sols,1),3,N-start)
    for i in range(1,size(sols,1))
        #extract timing
        _,_,_,_,guess,tot = extract_timing(sols[i],start)
        ret[i,1,:] = guess
        ret[i,2,:] = tot
        #extract GMRES iterations
        ret[i,3,:] = extract_iters(sols[i],start)
    end

    return ret
end

function Total_speedup_plot(fig,baseline,sols,sizes, tag)
    raw = Array{Float64}(undef,size(sols,1),size(sols,2))
    μ = Array{Float64}(undef,size(sols,2))
    σ = Array{Float64}(undef,size(sols,2))
    μ_GMRES = Array{Float64}(undef,size(sols,2))
    σ_GMRES = Array{Float64}(undef,size(sols,2))
    for i in range(1,size(sols,2))
        μ_GMRES[i] = mean(sum(baseline[:,i,3,:],dims=2) ./ sum(sols[:,i,3,:],dims=2))
        σ_GMRES[i] = std(sum(baseline[:,i,3,:],dims=2) ./ sum(sols[:,i,3,:],dims=2))
    end
    println(tag)
    println(sizes)
    println("GMRES: ", μ_GMRES)
    println("σ: ", σ_GMRES)

    for i in range(1,size(sols,2))
        raw[:,i] = sum(baseline[:,i,2,:],dims=2) ./ sum(sols[:,i,2,:],dims=2)
        μ[i] = mean(raw[:,i])
        σ[i] = std(raw[:,i])
    end

    guess = mean(sum(sols[:,:,1,:],dims=3)./ sum(sols[:,:,2,:],dims=3),dims=1)
    guess_σ = std(sum(sols[:,:,1,:],dims=3)./ sum(sols[:,:,2,:],dims=3),dims=1)
    println("Guess: ", guess)
    println("σ: ", guess_σ)

    scatter!(fig,sizes, μ, yerror=σ, markersize=7,label=tag,markerstrokecolor=:auto)
    return fig
end


method="LS"
dt = "10e_5"

#####
tt = pwd() * "/Examples/Data/Scaling/"*dt*"_"*method*"/scaling_Nystrom_33.jld2"
dat_tt = load(tt)
M = dat_tt["M"]
N = dat_tt["N"]
sizes = dat_tt["sizes"]
#####

start=33
stop=38
diff = stop - start + 1 
full_NYS = Array{Float64}(undef,diff,5,3,N-M)
full_QR = Array{Float64}(undef,diff,5,3,N-M)
full_SVD = Array{Float64}(undef,diff,5,3,N-M)
base = Array{Float64}(undef,diff,5,3,N-M)
for i in range(start,stop)
    filename_NYS = pwd() * "/Examples/Data/Scaling/"*dt*"_"*method*"/scaling_Nystrom_"*string(i)*".jld2"
    filename_QR = pwd() * "/Examples/Data/Scaling/"*dt*"_"*method*"/scaling_RQR_"*string(i)*".jld2"
    filename_SVD = pwd() * "/Examples/Data/Scaling/"*dt*"_"*method*"/scaling_RSVD_"*string(i)*".jld2"
    
    dat_NYS = load(filename_NYS)
    dat_QR = load(filename_QR)
    dat_SVD = load(filename_SVD)
    

    full_NYS[i-32,:,:,:]= extract(dat_NYS["sols"],M+1,N+1)
    full_QR[i-32,:,:,:] = extract(dat_QR["sols"],M+1,N+1)
    full_SVD[i-32,:,:,:] = extract(dat_SVD["sols"],M+1,N+1)
    base_NYS = extract(dat_NYS["base"],M+1,N+1)
    base_QR = extract(dat_QR["base"],M+1,N+1)
    base_SVD = extract(dat_SVD["base"],M+1,N+1)
    base[i-32,:,:,:] = (base_NYS + base_QR + base_SVD)/3

end
savefile = pwd() * "/Figures/Examples/Scaling/"*dt*"_"*method*"_seeds"



#Speedup as a function of the problem size
fig = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Problem size",
        ylabel="Total Speedup",
    )
Total_speedup_plot(fig,base,full_NYS,sizes,"Nyström")
Total_speedup_plot(fig,base,full_QR,sizes, "Range Finder")
Total_speedup_plot(fig,base,full_SVD,sizes, "Randomized SVD")

savefig(fig,savefile*".png")
