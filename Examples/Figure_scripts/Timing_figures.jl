using JLD2,Plots, LaTeXStrings, Statistics
include(pwd()*"/src/LSMOD.jl")
using .LSMOD

extract_iters(v,start) = [v[el][:history].niter for el in range(start,length(v))]

function extract_timing(data,start)
    precon = [data[el][:timing][:preconditioner] for el in range(start,length(data))]
    basis = [haskey(data[el][:timing],:basis) ? data[el][:timing][:basis] : 0.0 for el in range(start,length(data))]
    guess = [haskey(data[el][:timing],:guess) ? data[el][:timing][:guess] : 0.0 for el in range(start,length(data))]
    gmres = [data[el][:timing][:gmres] for el in range(start,length(data))]
    non_gmres = basis + guess
    tot = basis + guess + gmres 
    return precon,basis,guess,gmres, non_gmres,tot
end

function extract_reduced_timing(data,start)
    AQ = [data[el][:timing][:guess_detailed][:AQ] for el in range(start,length(data))]
    LS = [data[el][:timing][:guess_detailed][:LS] for el in range(start,length(data))]
    IG = [data[el][:timing][:guess_detailed][:IG] for el in range(start,length(data))]
    return AQ,LS,IG
end

function extract(base,start,N)
    ret = Array{Float64}(undef,6,N-start)
    #extract timing 
    _,basis,guess,_,_,tot = extract_timing(base,start)

    ret[1,:] = guess
    ret[2,:] = tot
    #extract GMRES iterations
    ret[3,:] = extract_iters(base,start)
    AQ,LS, _ = extract_reduced_timing(base,start)
    ret[4,:] = AQ 
    ret[5,:] = LS
    ret[6,:] = basis
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


function Total_speedup_plot(fig,fig2,baseline,sols,ind, tag)
    μ = Array{Float64}(undef,size(sols,2),3,size(sols,4))
    σ = Array{Float64}(undef,size(sols,2),3,size(sols,4))

    for i in range(1,size(sols,2))
        μ[i,:,:] = mean(cumsum(baseline,dims=3) ./ cumsum(sols[:,i,1:3,:],dims=3),dims=1) 
        σ[i,:,:] = std(cumsum(baseline,dims=3) ./ cumsum(sols[:,i,1:3,:],dims=3),dims=1)
        Plots.plot!(fig,ind,μ[i,2,ind], ribbon=σ[i,2,ind], linewidth=2, label=tag[i])
        Plots.scatter!(fig2,ind,mean(sols[:,i,2,ind],dims=1)[:], markerstrokecolor=:auto, label=tag[i])
        println(tags[i])
        println("Basis ", mean(mean(sols[:,i,6,:],dims=2))*1000, " ± ", std(mean(sols[:,i,6,:],dims=2))*1000)
        println("AQ ", mean(sols[:,i,4,:])*1000, " ± ", std(mean(sols[:,i,4,:],dims=2))*1000)
        println("LS ", mean(sols[:,i,5,:])*1000, " ± ", std(mean(sols[:,i,5,:],dims=2))*1000)
        println("GMRES ", mean(sum(baseline[:,3,:],dims=2) ./ sum(sols[:,i,3,:],dims=2)), " ± ", std(sum(baseline[:,3,:],dims=2) ./ sum(sols[:,i,3,:],dims=2)), " \n")
    end
    Plots.scatter!(fig2,ind,mean(baseline[:,2,ind],dims=1)[:], markerstrokecolor=:auto, label="Base")
    return fig
end



function reduced_time(data,M::Integer, title::String, ind)
    AQ,LS,IG = extract_reduced_timing(data,M)
    println("AQ: ", median(AQ), " LS: ", median(LS))
    println("AQ: ", median(AQ)/(4e-4), " LS: ", median(LS)/4e-4)
    #Proportion of each timing
    fig1 = scatter(ind, 
            [AQ[ind],LS[ind],IG[ind]], 
            title = title, 
            label= ["AQ" "Least squares" "Qs"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Time [s]")
    return fig1
end

function save_img(list,start,tag,file,ind)
    println(tag)
    f1,f2 = solver_time(list,start,ind)
    savefig(f1, file*"/"*tag*"_split.png")
    savefig(f2, file*"/"*tag*"_gmres.png")

    if tag!="base"
        f3 = reduced_time(list,start,tag,ind)
        savefig(f3, file*"/"*tag*"_guess.png")
    end
end

tstep = ARGS[1]
filename = pwd() * "/Examples/Data/"*tstep*"/final_33.jld2"
savefile = pwd() * "/Figures/Examples/Timing/"*tstep*"_seeds"


dat = load(filename)
M = dat["M"]
N = dat["N"]
step = 10
ind = (M+15):step:(N-M)
start=33
stop=38
diff = stop - start + 1

full = Array{Float64}(undef,diff,4,6,N-M)
full_base = Array{Float64}(undef,diff,3,N-M)
tags = ["Nystrom", "POD", "RandQR", "RandSVD"]
for i in range(start,stop)
    fn = pwd() * "/Examples/Data/"*tstep*"/final_"*string(i)*"_2.jld2"
    d = load(fn)
    for (j,tag) in enumerate(tags)
        full[i-start+1,j,:,:] = extract(d[tag],M+1,N+1)
    end
    full_base[i-start+1,:,:] = extract_base(d["base"],M+1,N+1)
end

#Speedup as a function of the problem size
fig = Plots.plot(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Iteration",
        ylabel="Speedup",
    )

fig2 = Plots.plot(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Iteration",
        ylabel="Total GMRES time [s]",
    )
plot_tags = ["Nyström", "POD", "Range Finder", "Randomized SVD"]
Total_speedup_plot(fig,fig2,full_base,full,ind,plot_tags)
display(fig)
savefig(fig,savefile*"/speedup_"*tstep*"_2.png")
savefig(fig2,savefile*"/vals_"*tstep*"_2.png")

