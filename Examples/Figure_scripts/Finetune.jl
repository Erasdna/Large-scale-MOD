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
    ret = Array{Float64}(undef,size(sols,1),size(sols,2),3,N-start)
    for i in range(1,size(sols,1))
        #extract timing
        for j in range(1,size(sols,2))
            _,_,_,_,guess,tot = extract_timing(sols[i,j],start)
            ret[i,j,1,:] = guess
            ret[i,j,2,:] = tot
            #extract GMRES iterations
            ret[i,j,3,:] = extract_iters(sols[i,j],start)
        end
    end

    return ret
end

function extract_base(base,start,N)
    ret = Array{Float64}(undef,3,N-start)
    #extract timing 
    _,_,guess,tot,_,_ = extract_timing(base,start)

    ret[1,:] = guess
    ret[2,:] = tot
    #extract GMRES iterations
    ret[3,:] = extract_iters(base,start)

    return ret
end

function Total_speedup_plot(fig,baseline,sols,m_frac,Ms)
    raw = Array{Float64}(undef,size(sols,2),size(sols,3))
    σ =  Array{Float64}(undef,size(sols,2),size(sols,3))
    GMRES = Array{Float64}(undef,size(sols,2),size(sols,3))

    for i in range(1,size(sols,2))
        for j in range(1,size(sols,3))
            GMRES[i,j] = mean(sum(baseline[:,3,:],dims=2) ./ sum(sols[:,i,j,3,:],dims=2))
        end
        println(Ms[i], ": ", GMRES[i,:])
    end

    for i in range(1,size(sols,2))
        for j in range(1,size(sols,3))
            raw[i,j] = mean(sum(baseline[:,2,:],dims=2) ./ sum(sols[:,i,j,2,:],dims=2))
            σ[i,j] = std(sum(baseline[:,2,:],dims=2) ./ sum(sols[:,i,j,2,:],dims=2))
        end
        scatter!(fig,m_frac,raw[i,:],yerror = σ, markersize=7,label="M="*string(Ms[i]),markerstrokecolor=:auto)
    end

    return fig
end

dt = "10e_3"
filename = pwd() * "/Examples/Data/Finetune/"*dt*"_N=200/finetune_Nystrom_LS_33.jld2"
savefile = pwd() * "/Figures/Examples/Finetune/"*dt*"_finetune_Nystrom_LS_seeds"

####
dd = load(filename)
M = dd["M"]
N = dd["N"]
m_frac = dd["m"]
####

start=33
stop=38
Mm = maximum(M)
diff = stop - start + 1 
full = Array{Float64}(undef,diff,3,4,3,N-Mm)
full_base = Array{Float64}(undef,diff,3,N-Mm)

for i in range(start,stop)
    fn = pwd() * "/Examples/Data/Finetune/"*dt*"_N=200/finetune_Nystrom_LS_"*string(i)*".jld2"
    dat = load(fn)
    full[i-start+1,:,:,:,:] = extract(dat["sols"],Mm+1,N+1)
    full_base[i-start+1,:,:] = extract_base(dat["base"],Mm+1,N+1)
end

#Speedup as a function of the problem size
fig = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="m/M",
        ylabel="Total Speedup",
    )
Total_speedup_plot(fig,full_base,full,m_frac,M)
savefig(fig,savefile*".png")