@inline function numerical_Zhat(Mu_mc, Sig_mc, g₀, ℓ, t_max) 
    n_mc = size(Mu_mc, 3) 
    gg = zeros(Float64, t_max, n_mc)

    @inbounds for n = 1:n_mc, k = ℓ:(ℓ+t_max-1)  
        min_d = 1.
        for i = 1:(k-1), j = (i+1):k
            d = wass_gauss(
                Mu_mc[:, i, n], Sig_mc[:, :, i, n], 
                Mu_mc[:, j, n], Sig_mc[:, :, j, n]) 
            min_d = min(min_d, d/(g₀+d)) 
        end
        gg[k-ℓ+1, n] = min_d  
    end 
    Ẑ = log.(mean(gg; dims=2))
    return Ẑ
end 