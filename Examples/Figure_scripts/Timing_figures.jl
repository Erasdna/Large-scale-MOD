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

function solver_time(data,M::Integer,ind)
    precon,basis,guess,gmres,non_gmres,_ = extract_timing(data,M)
    iters = extract_iters(data,M)
    println("basis: ", median(basis))
    println("basis: ", median(basis)/4e-4)
    println("GMRES time per iteration: ", sum(gmres)/sum(iters))
    #Proportion of each timing
    fig1 = scatter(ind, 
            [cumsum(precon)[ind],cumsum(basis)[ind],cumsum(guess)[ind],cumsum(gmres)[ind]], 
            label= ["Preconditioner" "Basis" "Guess" "GMRES"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Cumulative time [s]")
    fig2 = scatter(ind, 
            [non_gmres[ind],gmres[ind]], 
            label= ["Initial guess" "GMRES"], 
            lw=2,
            xlabel="Timestep",
            ylabel="Cumulative time [s]")
    return fig1,fig2
end

function method_comp(base,method1, tag1::String, method2, tag2::String, method3,tag3,method4,tag4,M,ind)
    b = extract_timing(base,M)
    b_iters = extract_iters(base,M)
    m1 = extract_timing(method1,M)
    m1_iters = extract_iters(method1,M)
    println(tag1, " ", sum(b_iters)/sum(m1_iters))
    m2 = extract_timing(method2,M)
    m2_iters = extract_iters(method2,M)
    println(tag2, " ", sum(b_iters)/sum(m2_iters))
    m3 = extract_timing(method3,M)
    m3_iters = extract_iters(method3,M)
    println(tag3, " ", sum(b_iters)/sum(m3_iters))
    m4 = extract_timing(method4,M)
    m4_iters = extract_iters(method4,M)
    println(tag4, " ", sum(b_iters)/sum(m4_iters))
    fig1 = scatter(ind, 
            [m1[end][ind],m2[end][ind],m3[end][ind],m4[end][ind],b[end][ind]], 
            label= [tag1 tag2 tag3 tag4 "base"], 
            lw=2,
            guidefontsize=14,
            tickfontsize=12,
            legendfontsize=12,
            ylims=(0,maximum(b[end][ind])+0.001),
            xlabel="Timestep",
            ylabel="Time [s]")
    bb=cumsum(b[end])
    fig2 = scatter(ind, 
            [(bb./cumsum(m1[end]))[ind],(bb./cumsum(m2[end]))[ind],(bb./cumsum(m3[end]))[ind],(bb./cumsum(m4[end]))[ind]], 
            label= [tag1 tag2 tag3 tag4], 
            lw=2,
            guidefontsize=14,
            tickfontsize=12,
            legendfontsize=12,
            xlabel="Timestep",
            ylabel="Speedup")
    return fig1,fig2
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

filename = pwd() * "/Examples/Data/10e_3_update.jld2"
savefile = pwd() * "/Figures/Examples/Timing/10e_3_update"
dat = load(filename)

start = dat["M"]+2
step = 10
ind = 1:step:(dat["N"]-start)

save_img(dat["base"],start,"base",savefile,ind)
save_img(dat["POD"],start,"POD",savefile,ind)
save_img(dat["RandQR"],start,"RangeFinder",savefile,ind)
save_img(dat["RandSVD"],start,"RandomizedSVD",savefile,ind)
save_img(dat["Nystrom"],start,"Nystrom",savefile,ind)

f3,f4 = method_comp(dat["base"],dat["Nystrom"],"Nystrom",dat["POD"],"POD",dat["RandQR"],"RangeFinder", dat["RandSVD"], "Randomized SVD", start,ind)
savefig(f3, savefile*"/comp.png")
savefig(f3, savefile*"/comp.svg")
savefig(f4, savefile*"/cumulative_comp.png")
savefig(f4, savefile*"/cumulative_comp.svg")