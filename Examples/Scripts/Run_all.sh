# !/bin/sh

sizes="10e_3 10e_5"

#Iterate running all methods over seeds

for seed in $(seq 33 35);
do
    echo $seed
    for tstep in $sizes;
    do
        echo $tstep
        julia --project=. Examples/Experiments/Run_all.jl $tstep $seed
    done
done
