using JLD2,Plots, LaTeXStrings
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

function extract_reduced_timing(data,start)
    AQ = [data[el][:timing][:guess_detailed][:AQ] for el in range(start,length(data))]
    LS = [data[el][:timing][:guess_detailed][:LS] for el in range(start,length(data))]
    IG = [data[el][:timing][:guess_detailed][:IG] for el in range(start,length(data))]
    return AQ,LS,IG
end

function solver_time(data,M::Integer,ind)
    precon,basis,guess,gmres,non_gmres,_ = extract_timing(data,M)

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
    m1 = extract_timing(method1,M)
    m2 = extract_timing(method2,M)
    m3 = extract_timing(method3,M)
    m4 = extract_timing(method4,M)
    fig1 = scatter(ind, 
            [b[end][ind],m1[end][ind],m2[end][ind],m3[end][ind],m4[end][ind]], 
            label= ["base" tag1 tag2 tag3 tag4], 
            lw=2,
            xlabel="Timestep",
            ylabel="Time [s]")
    bb=cumsum(b[end])
    fig2 = scatter(ind, 
            [(bb./cumsum(m1[end]))[ind],(bb./cumsum(m2[end]))[ind],(bb./cumsum(m3[end]))[ind],(bb./cumsum(m4[end]))[ind]], 
            label= [tag1 tag2 tag3 tag4], 
            lw=2,
            xlabel="Timestep",
            ylabel="Speedup")
    return fig1,fig2
end

function reduced_time(data,M::Integer, title::String, ind)
    AQ,LS,IG = extract_reduced_timing(data,M)

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
    f1,f2 = solver_time(list,start,ind)
    savefig(f1, file*"/"*tag*"_split.png")
    savefig(f2, file*"/"*tag*"_gmres.png")

    if tag!="base"
        f3 = reduced_time(list,start,tag,ind)
        savefig(f3, file*"/"*tag*"_guess.png")
    end
end

filename = pwd() * "/Examples/Data/10e_5_all_noproj.jld2"
savefile = pwd() * "/Figures/Examples/Timing/10e_5_all"
dat = load(filename)

start = dat["M"]+1
step = 10
ind = 1:step:(dat["N"]-start)

save_img(dat["base"],start,"base",savefile,ind)
save_img(dat["POD"],start,"POD",savefile,ind)
save_img(dat["RandQR"],start,"RangeFinder",savefile,ind)
save_img(dat["RandSVD"],start,"RandomizedSVD",savefile,ind)
save_img(dat["Nystrom"],start,"Nystrom",savefile,ind)

f3,f4 = method_comp(dat["base"],dat["POD"],"POD",dat["RandQR"],"RangeFinder", dat["RandSVD"], "Randomized SVD", dat["Nystrom"],"Nystrom",start,ind)
savefig(f3, savefile*"/comp.png")
savefig(f3, savefile*"/comp.svg")
savefig(f4, savefile*"/cumulative_comp.png")
savefig(f4, savefile*"/cumulative_comp.svg")