# Numerical Computation of ZK as in algorithm 1
@inline function numerical_Zₖ(K_max, dim, config, k_prior; n_mc=1000)
    g₀ = config["g₀"]
    a₀ = config["a₀"]
    b₀ = config["b₀"]
    τ = config["τ"]
    method = config["method"]
    
    !occursin("dpgm", method) || return ones(K_max)

    μ_mc = zeros(dim, K_max, n_mc)
    Σ_mc = zeros(dim, dim, K_max, n_mc)
     
    indices = CartesianIndices((1:K_max, 1:n_mc))
    Threads.@threads for (k, n) in Tuple.(indices) 
        μ, Σ = rand(k_prior)
        μ_mc[:, k, n] .= μ
        Σ_mc[:, :, k, n] .= Σ 
    end 

    dist_fn = @match method begin 
        "wrgm-full" || "wrgm-diag"  => wass_dist_gauss 
        "rgm-full" || "rgm-diag"    => mean_dist_gauss 
    end  

    gg = zeros(K_max, n_mc) 
    @showprogress Threads.@threads for n = 1:n_mc
        for k = 2:K_max 
            min_d = 1.
            @inbounds for i = 1:(k - 1), j = (i + 1):k 
                d = dist_fn(
                    μ_mc[:, i, n], Σ_mc[:, :, i, n], 
                    μ_mc[:, j, n], Σ_mc[:, :, j, n]) 
                min_d = min(min_d, d/(g₀+d))
            end
            gg[k, n] = min_d  
        end
    end 

    Z = log.(mean(gg; dims=2)) |> vec
    Z[1] = 0
    return Z
end