module WRGM

export wrbgmm_blocked_gibbs, MCSample
export KernelPrior, DiagPrior, FullPrior
export mean_dist_gauss, wass_dist_gauss, min_distance 
 
using Distributions 
using FStrings
using LinearAlgebra 
using LoopVectorization
using MLStyle
using ProgressMeter
using Random 
using SpecialFunctions
using Statistics
using StatsBase 

import Base.rand 

include("log_p_K.jl")
include("logV.jl")
include("numerical_Zk.jl")
include("numerical_Zhat.jl")

include("measure/distance.jl")
include("mcmc/kernel_prior.jl")
include("mcmc/blocked_gibbs.jl") 


logpdf = Distributions.logpdf

const GUMBEL = Gumbel(0, 1)

rand_categorical(logits) = argmax(logits + rand(GUMBEL, length(logits))) 

end # module WassersteinRepulsiveGM
