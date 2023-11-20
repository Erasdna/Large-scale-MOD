#! /bin/sh

julia --project=. Examples/LS_examples.jl RQR 10e_3
julia --project=. Examples/LS_examples.jl RQR 10e_5
julia --project=. Examples/LS_examples.jl Nystrom 10e_3
julia --project=. Examples/LS_examples.jl Nystrom 10e_5