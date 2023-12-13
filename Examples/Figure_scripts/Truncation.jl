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

function Total_speedup_plot(baseline,sols,p)
    raw = Array{Float64}(undef,size(sols,1))
    raw = sum(baseline[1,2,:]) ./ sum(sols[:,2,:],dims=2)
    
    guess = sum(sols[:,1,:],dims=2)
    println("Guess: ", guess)
    GMRES = sum(baseline[1,3,:]) ./ sum(sols[:,3,:],dims=2)
    println("GMRES: ", GMRES)
    fig1 = scatter(
        lw=2,
        guidefontsize=14,
        tickfontsize=12,
        legendfontsize=12,
        xlabel="Oversampling parameter",
        ylabel="Total Speedup",
    )
    scatter!(fig1,p .- 1, raw[:,1])

    return fig1
end

filename = pwd() * "/Examples/Data/Truncation/10e_5_truncation_RSVD_33.jld2"
filename2 = pwd() * "/Examples/Data/Truncation/10e_5_truncation_RSVD_35.jld2"

savefile = pwd() * "/Figures/Examples/Truncation/10e_5_truncation_RSVD_33"
dat = load(filename)
dat2 = load(filename2)
M = dat["M"]
N = dat["N"]
m = dat["m"]
pf = dat["pf"]

mat1 = extract(dat["sols"],M+1,N+1)
mat2 = extract(dat2["sols"],M+1,N+1)
mat = (mat1 .+ mat2)./2
base1 = extract_base(dat["base"],M+1,N+1)
base2 = extract_base(dat2["base"],M+1,N+1)
base = (base1 .+ base2)./2
fig1 = Total_speedup_plot(base, mat,collect(range(1,pf)))
savefig(fig1,savefile*".png")