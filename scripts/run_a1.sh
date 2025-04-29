#!/bin/bash
set -e

data="a1" 
n_burnin=10000
n_iter=5000 
tau=1e4
g0=100
nu0=4
thinning=2

 
# for method in "rgm-full" "rgm-diag" "wrgm-full" "wrgm-diag" "mfm-full" "mfm-diag" 
# do 
# julia -t4 ./test/run.jl \
# 	--dataname $data \
# 	--method $method \
# 	--n_burnin $n_burnin \
# 	--n_iter $n_iter \
# 	--thinning $thinning \
# 	--tau $tau \
# 	--g0 $g0 \
# 	--nu0 $nu0  
# done 
# ##################################################################################


# for method in "rgm-full" "rgm-diag" "wrgm-full" "wrgm-diag" "mfm-full" "mfm-diag" 
# do 
# 	julia -t8 ./test/plot.jl --dataname $data --method $method  
# done
# ##################################################################################


# julia -t8 ./test/plot.jl --dataname $data --method all 

julia -t8 ./test/plot.jl --dataname $data --method all --dist_type Mean
julia -t8 ./test/plot.jl --dataname $data --method all --dist_type Wasserstein
