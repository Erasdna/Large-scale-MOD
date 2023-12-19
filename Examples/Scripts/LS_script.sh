#! /bin/sh

#Run row sampling test over method and timestep. 
#Takes seed as input

methods="Nystrom RQR RSVD"
timestep="10e_3 10e_5"

for m in $methods;
do
    for t in $timestep;
    do 
        julia --project=. Examples/Experiments/LS_examples.jl $m $t $1
    done
done