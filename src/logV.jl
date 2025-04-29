@inline function logV_nt(n, β, K; λ=1) 
    log_V = zeros(K)
    tol = 1e-12 
 
    log_pk(k) = k * log(λ) - log(exp(λ) - 1) - logfactorial(k) 
    @inbounds for t = 1:K
        log_V[t] = -Inf
        if t <= n 
            a, c, k, p = 0, -Inf, 1, 0
            while abs(a - c) > tol || p < 1.0 - tol
                # Note: The first condition is false when a = c = -Inf
                if k >= t
                    a = c  
                    b = loggamma(k+1) - loggamma(k-t+1) 
                    b += - loggamma(β*k+n) + loggamma(β*k) 
                    b += log_pk(k)
                    c = logsumexp(a, b) 
                end
                p += exp(log_pk(k))
                k += 1 
            end
            log_V[t] = c
        end 
    end 
    return log_V
end
 

logsumexp(a, b) = begin
    m = max(a, b)
    m == -Inf ? -Inf : log(exp(a-m) + exp(b-m)) + m
end 
 