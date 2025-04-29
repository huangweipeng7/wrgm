mean_dist_gauss(μ₁, Σ₁, μ₂, Σ₂) = sum(@turbo (μ₁ .- μ₂) .^ 2) |> sqrt


@inline function wass_dist_gauss(μ₁, Σ₁, μ₂, Σ₂) 
    Σ₂_sqrt = sqrt(Σ₂) 
    Σ_part_sqrt = 2 * sqrt(Σ₂_sqrt * Σ₁ * Σ₂_sqrt)   
    μ_part = @tturbo (μ₁ .- μ₂) .^ 2 
    Σ_part = @tturbo Σ₁ .+ Σ₂ .- Σ_part_sqrt 
    return sqrt(sum(μ_part) + tr(Σ_part)) 
end


function min_distance(Mu, Sig, g₀, method)
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    (occursin("mfm", method) && g₀ ≠ 0) &&
        raise("For a no-repulsion setting, g₀ ≠ 0.")

    g₀ != 0 || return 1.0

    dist_fn = @match method begin 
        "wrgm-full" || "wrgm-diag"  => wass_dist_gauss 
        "rgm-full" || "rgm-diag"    => mean_dist_gauss 
    end  

    hₖ = 1.0
    K = size(Mu, 2)
    d_mat = ones(K, K)
  
    indices = filter(c -> c[1] < c[2], CartesianIndices((1:K, 1:K)))
    Threads.@threads for (i, j) in Tuple.(indices)
        @inbounds d = dist_fn(
            Mu[:, i], Sig[:, :, i], Mu[:, j], Sig[:, :, j])   
        d_mat[i, j] = d / (d + g₀) 
    end   
    return minimum(d_mat)
end 