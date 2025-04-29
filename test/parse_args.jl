using ArgParse 


function parse_cmd()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--dataname"
            help = "data name"
            arg_type = String 
            required = true  
        "--method"
            help = "method name: either mean, wasserstein, or no-rep"
            arg_type = String
            required = true  
        "--n_burnin"
            help = "number of burn in iterations"
            arg_type = Int 
            required = true  
        "--n_iter"
            help = "number of iterations after burn in"
            arg_type = Int
            required = true  
        "--thinning"
            help = "thinning in the MCMC"
            arg_type = Int
            required = true  
        "--tau" 
            help = "scale factor for the diagonal matrix in the prior Normal for μ"
            arg_type = Float64 
            required = true
        "--nu0"
            help = "degree of freedom in the inverse Wishart"
            arg_type = Float64 
            required = true 
        "--g0"
            help = "g₀ for the hₖ computation"
            arg_type = Float64 
            required = true  
    end
 
    return parse_args(s)
end


function parse_plot_cmd()
    s = ArgParseSettings()

    @add_arg_table s begin 
        "--dataname"
            help = "data name"
            arg_type = String 
            required = true  
        "--method"
            help = "method name: either mean, wasserstein, or none"
            arg_type = String
            required = true  
        "--dist_type"
            help = "distance type: either mean or wasserstein"
            arg_type = String
            required = false  
    end
 
    return parse_args(s)
end