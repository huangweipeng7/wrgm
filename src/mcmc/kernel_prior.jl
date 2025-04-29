abstract type KernelPrior end 


@inline function post_sample_gauss!(
    X, Mu, Sig, C, k_prior::KernelPrior)

    K = size(Mu, 2)
    @inbounds for k in 1:K
        Xₖ = X[:, C .== k] 
        n = size(Xₖ, 2)  
        
        μ, Σ = n == 0 ? 
            rand(k_prior) : 
            post_sample_gauss(Xₖ, Mu[:, :, k], Sig[:, :, k], k_prior)
         
        Mu[:, k] .= μ[:] 
        Sig[:, :, k] .= Σ[:, :]
    end 
end 


############################################################ 

struct ScaleMvNormal
    τ::Float64
    mvn::MvNormal
end 


struct EigBoundedIW
    l_σ2::Float64
    u_σ2::Float64
    iw::InverseWishart 
end 


struct FullPrior <: KernelPrior
    dim::Int
    smvn::ScaleMvNormal
    biw::EigBoundedIW
end 


EigBoundedIW(l_σ2, u_σ2, ν₀, Φ₀) = EigBoundedIW(
    l_σ2, u_σ2, InverseWishart(ν₀, round.(Φ₀, digits=10) |> Matrix))


FullPrior(τ, μ₀, Σ₀, l_σ2, u_σ2, ν₀, Φ₀) = FullPrior(
    size(μ₀, 1), 
    ScaleMvNormal(τ, MvNormal(μ₀, Σ₀)), 
    EigBoundedIW(l_σ2, u_σ2, ν₀, round.(Φ₀, digits=10) |> Matrix)) 


clogpdf(prior::FullPrior, μ, Σ) =
    logpdf(prior.smvn.mvn, μ) + logpdf(prior.biw.iw, Σ)


@inline function rand(biw::EigBoundedIW; max_cnt=2000, approx=true)  
    dim = size(biw.iw.Ψ, 1) 
    
    Σ = zeros(dim, dim)
    if !approx
        eig_v_Σ = nothing 

        l_σ2, u_σ2 = biw.l_σ2, biw.u_σ2
        @inbounds for c = 1:max_cnt 
            Σ .= rand(biw.iw) 
            Σ .= round.(Σ, digits=10)    
            eig_v_Σ = eigvals(Σ)
            
            (first(eig_v_Σ) > l_σ2 && last(eig_v_Σ) < u_σ2) && break  
        end  
        throw(
            "Sampling from the prior takes too long. 
            Check if the bounds are set properly")
    else 
        Σ .= rand(biw.iw) 
    end 
    return Σ
end 


function rand(prior::FullPrior; max_cnt=2000)  
    Σ = rand(prior.biw; max_cnt=max_cnt)
    μ = rand(prior.smvn.mvn)   
    return μ, Σ
end 



@inline function post_sample_gauss(X, μ, Σ, k_prior::FullPrior)
    """ A function for the posterior sampling of Gaussian kernels.
        μ is currently not in use.
    """
    # An inner function for computing the covariance
    cov_(X) = cov(X; dims=2, corrected=false) * n  

    dim, n = size(X) 
  
    x̄ = mean(X; dims=2)

    νₙ = k_prior.biw.iw.df + n 
    Ψₙ = Matrix(k_prior.biw.iw.Ψ) + cov_(X) 
    Ψₙ = round.(Ψₙ, digits=8)
    biw_p = EigBoundedIW(k_prior.biw.l_σ2, k_prior.biw.u_σ2, νₙ, Ψₙ)
    Σ = rand(biw_p)

    Σ₀ = inv(inv(k_prior.smvn.mvn.Σ) + n * inv(Σ))
    Σ₀ = round.(Σ₀, digits=8)
    μ₀ = Σ₀ * (n * inv(Σ) * x̄) |> vec 
    normal = try 
        MvNormal(μ₀, Σ₀) 
    catch LoadError
        # Most likely there are some numerical inconsistencies of  
        # the decimal values in the non-diagonal elements 
        Σ₀[1, 2] = Σ₀[2, 1]; # Only works for 2-dim data
                             # round() here may sometimes fail
        MvNormal(μ₀, Σ₀)  
    end 
    μ = rand(normal)

    return μ, Σ
end 


##################################################

mutable struct DiagPrior <: KernelPrior
    dim::Int
    a::Real
    b::Vector{Real}
    l_σ2::Real
    u_σ2::Real
    τ::Real
end 

DiagPrior(dim, a, b::Real, l_σ2, u_σ2, τ) = 
    DiagPrior(dim, a, fill(b, dim), l_σ2, u_σ2, τ)


function rand(k_prior::DiagPrior)  
    μ = randn(k_prior.dim) * k_prior.τ
    Σ = rand_inv_gamma(k_prior)   
    return μ, Σ
end 


clogpdf(k_prior::DiagPrior, μ, Σ) =
    logpdf(MvNormal(zeros(k_prior.dim), k_prior.τ^2*I(k_prior.dim)), μ) +
    sum(logpdf(InverseGamma(k_prior.a, k_prior.b[i]), Σ[i, i]) for i in 1:k_prior.dim)


function post_sample_gauss(X, μ, Σ, k_prior::DiagPrior)
    dim, n = size(X)
    τ = k_prior.τ

    x_sum = sum(X; dims=2)  
    Σ₀ = inv(τ^(-2)*I + n * inv(Σ))
    μ₀ = Σ₀ * (inv(Σ) * x_sum) |> vec 
    μ = rand(MvNormal(μ₀, Σ₀))

    aₖ = k_prior.a + n / 2 
    bₖ = (k_prior.b .+ sum((X .- μ).^2; dims=2) / 2) |> vec
    prior_p = deepcopy(k_prior)
    prior_p.a, prior_p.b = aₖ, bₖ
    Σ = rand_inv_gamma(prior_p) 

    return μ, Σ
end 


@inline function rand_inv_gamma(k_prior::DiagPrior; n=1) 
    dim = k_prior.dim
    l_σ2 = k_prior.l_σ2
    u_σ2 = k_prior.u_σ2
    a = k_prior.a 
    b = k_prior.b 

    Λ = n == 1 ? Diagonal(zeros(Float64, dim)) : zeros(Float64, dim, dim, n) 
    @inbounds for p in 1:dim
        if n == 1
            # Using gamma to sample inverse gamma R.V. is always more robust in Julia
            σ2 = truncated(Gamma(a, 1/b[p]), 1/u_σ2, 1/l_σ2) |> rand 
            @assert σ2 > 0
            Λ[p, p] = 1 / σ2
        else 
            # Using gamma to sample inverse gamma R.V. is always more robust in Julia
            σ2 = rand(truncated(Gamma(a, 1/b[p]), 1/u_σ2, 1/l_σ2), n) 
            @assert all(σ2 .> 0)
            Λ[p, p, :] .= 1 ./ σ2
        end  
    end  
    return Λ
end 
