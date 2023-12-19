#! /bin/sh

#Run row sampling scaling test over method and timestep. 
#Takes seed as input

methods="Nystrom RQR RSVD"
timestep="10e_3 10e_5"

for m in $methods;
do
    for t in $timestep;
    do 
        julia --project=. Examples/Experiments/Scaling_example.jl $m $t LS $1
    done
done
