@inline compute_log_prob(k::Int, ℓ::Int, n::Int) = 
	logfactorial(k) - logfactorial(k-ℓ) - logfactorial(k+n)


@inline log_prob_K(ℓ::Int, t_max::Int, n::Int) = 
	compute_log_prob.(ℓ:ℓ+t_max-1, Ref(ℓ), Ref(n)) 