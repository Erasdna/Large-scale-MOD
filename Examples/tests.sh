# !/bin/sh

sizes="10e_3 10e_5"
method="LS no"

for seed in $(seq 33 43);
do
    echo $seed
    for tstep in $sizes;
    do
        echo $tstep
        for m in $method;
        do
            echo $m
            julia --project=. Examples/Scaling_example.jl Nystrom $tstep $m $seed
            julia --project=. Examples/Scaling_example.jl RQR $tstep $m $seed
            julia --project=. Examples/Scaling_example.jl RSVD $tstep $m $seed
        done
        julia --project=. Examples/Finetune_example.jl Nystrom $tstep LS 200 $seed
    done
done
