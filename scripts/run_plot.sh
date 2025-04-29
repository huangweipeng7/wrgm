for data in "sim_data1" "sim_data2" "sim_data3" "a1" "GvHD" 
do 
	for method in "rgm-full" "rgm-diag" "wrgm-full" "wrgm-diag" "dpgm-full" "dpgm-diag" 
	do 
		julia -t8 ./test/plot.jl --dataname $data --method $method  
	done
	##################################################################################

	julia -t8 ./test/plot.jl --dataname $data --method "true"
	julia -t8 ./test/plot.jl --dataname $data --method all --dist_type Mean
	julia -t8 ./test/plot.jl --dataname $data --method all --dist_type Wasserstein
done 


# for data in "sim_data1" "sim_data2" "sim_data3" 
# do 
# 	for method in "rgm-full" "wrgm-full" "dpgm-full" 
# 	do 
# 		julia -t8 ./test/plot.jl --dataname $data --method $method  
# 	done
# 	julia -t8 ./test/plot.jl --dataname $data --method "true"
# 	julia -t8 ./test/plot.jl --dataname $data --method all --dist_type Mean
# 	julia -t8 ./test/plot.jl --dataname $data --method all --dist_type Wasserstein
# done 