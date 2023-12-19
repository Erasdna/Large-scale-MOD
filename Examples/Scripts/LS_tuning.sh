# !/bin/sh

sizes="10e_3 10e_5"

for seed in $(seq 33 43);
do
    echo $seed
    for tstep in $sizes;
    do
        echo $tstep
        julia --project=. Examples/Experiments/LS_examples.jl Nystrom $tstep $seed
        julia --project=. Examples/Experiments/LS_examples.jl RQR $tstep $seed
        julia --project=. Examples/Experiments/LS_examples.jl RSVD $tstep $seed
        julia --project=. Examples/Finetune_example.jl Nystrom $tstep no 100 $seed
    done
done
