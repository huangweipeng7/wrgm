#!/bin/bash
set -e

data="GvHD" 
n_burnin=10000 
n_iter=5000 
tau=100
g0=50
nu0=4
thinning=1 
  
  
for method in "rgm-full" "rgm-diag" "wrgm-full" "wrgm-diag" "mfm-full" "mfm-diag" 
do 
julia -t4 ./test/run.jl \
	--dataname $data \
	--method $method \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0  
done 
##################################################################################


for method in "rgm-full" "rgm-diag" "wrgm-full" "wrgm-diag" "mfm-full" "mfm-diag" 
do 
	julia -t8 ./test/plot.jl --dataname $data --method $method  
done
##################################################################################


julia -t8 ./test/plot.jl --dataname $data --method all 