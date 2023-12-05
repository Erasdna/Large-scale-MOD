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

function extract_base(base,start,N)
    ret = Array{Float64}(undef,1,3,N-start)
    #extract timing 
    _,_,guess,tot,_,_ = extract_timing(base,start)
    ret[1,1,:] = guess
    ret[1,2,:] = tot
    #extract GMRES iterations
    ret[1,3,:] = extract_iters(base,start)

    return ret
end

function Total_speedup_plot(fig,baseline,sols,sizes, tag)
    raw = Array{Float64}(undef,size(sols,1))
    GMRES = Array{Float64}(undef,size(sols,1))
    for i in range(1,size(baseline,1))
        GMRES[i] = sum(baseline[i,3,:]) ./ sum(sols[i,3,:])
    end
    println(tag)
    println(sizes)
    println("GMRES: ", GMRES)

    for i in range(1,size(baseline,1))
        raw[i] = sum(baseline[i,2,:]) ./ sum(sols[i,2,:])
    end

    guess = sum(sols[:,1,:],dims=2)./ sum(sols[:,2,:],dims=2)
    println("Guess: ", guess)

    scatter!(fig,sizes, raw,markersize=7,label=tag)
    return fig
end

function Split_speedup_scatter(fig,baseline,sols,tag,color)

    guess = sum(sols[:,1,:],dims=2)
    println("Guess: ", guess)
    GMRES = Array{Float64}(undef,size(sols,1))
    for i in range(1,size(baseline,1))
        GMRES[i] = sum(sols[i,3,:]) ./ sum(baseline[i,3,:])
    end
    println("GMRES: ", GMRES)
    shapes = [:star,:diamond,:circle,:hexagon, :ltriangle]
    scatter!(fig,guess, GMRES,markershape=shapes,color=color,label=tag)
end

dt = "10e_5"
filename_NYS = pwd() * "/Examples/Data/Scaling/"*dt*"_scaling_Nystrom_LS.jld2"
filename_QR = pwd() * "/Examples/Data/Scaling/"*dt*"_scaling_RQR_LS.jld2"
filename_SVD = pwd() * "/Examples/Data/Scaling/"*dt*"_scaling_RSVD_LS.jld2"
savefile = pwd() * "/Figures/Examples/Scaling/"*dt*"_scaling_comp_LS"

dat_NYS = load(filename_NYS)
dat_QR = load(filename_QR)
dat_SVD = load(filename_SVD)
M = dat_NYS["M"]
N = dat_NYS["N"]
sizes = dat_NYS["sizes"]
ind = [1,2]

mat_NYS = extract(dat_NYS["sols"],M+1,N+1)
mat_QR = extract(dat_QR["sols"],M+1,N+1)
mat_SVD = extract(dat_SVD["sols"],M+1,N+1)


base_NYS = extract(dat_NYS["base"],M+1,N+1)
base_QR = extract(dat_QR["base"],M+1,N+1)
base_SVD = extract(dat_SVD["base"],M+1,N+1)

#Speedup as a function of the problem size
fig = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Problem size",
        ylabel="Total Speedup",
    )
Total_speedup_plot(fig,base_NYS,mat_NYS,sizes,"Nyström")
Total_speedup_plot(fig,base_QR,mat_QR,sizes, "Range Finder")
Total_speedup_plot(fig,base_SVD,mat_SVD,sizes, "Randomized SVD")

savefig(fig,savefile*".png")

# fig2 = scatter(
#         lw=2,
#         guidefontsize=14,
#         tickfontsize=12,
#         legendfontsize=12,
#         xlabel="Initial guess speedup",
#         ylabel="GMRES speedup",
#     )

# Split_speedup_scatter(fig2,base_NYS,mat_NYS,"Nyström",:blue)
# Split_speedup_scatter(fig2,base_QR,mat_QR,"Range Finder", :green)
# Split_speedup_scatter(fig2,base_SVD,mat_SVD, "Randomized SVD", :purple)