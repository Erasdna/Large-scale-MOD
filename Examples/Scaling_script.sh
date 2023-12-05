#! /bin/sh

julia --project=. Examples/Scaling_example.jl Nystrom 10e_3 LS
julia --project=. Examples/Scaling_example.jl RQR 10e_3 LS
julia --project=. Examples/Scaling_example.jl RSVD 10e_3 LS

julia --project=. Examples/Scaling_example.jl Nystrom 10e_5 LS
julia --project=. Examples/Scaling_example.jl RQR 10e_5 LS
julia --project=. Examples/Scaling_example.jl RSVD 10e_5 LS