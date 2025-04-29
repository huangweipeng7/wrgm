using OrderedCollections
using DataFrames 
using Gadfly  
using Plots, StatsPlots
using LinearAlgebra   
using JLD2   
using MLStyle
using Distributions, StatsBase, StatsFuns
using WRGM 
 
import PlotlyKaleido    
import ColorSchemes as cs 

include("./data.jl"); import .load_data
include("./parse_args.jl"); import .parse_plot_cmd
 

function wass_dist(μ₁, Σ₁, μ₂, Σ₂) 
    Σ₂_sqrt = sqrt(Σ₂) 
    Σ_part_sqrt = 2 * sqrt(Float64.(Σ₂_sqrt * Σ₁ * Σ₂_sqrt))   
    μ_part = @. (μ₁ - μ₂) ^ 2 
    Σ_part = @. Σ₁ + Σ₂ - Σ_part_sqrt 
    return sqrt(sum(μ_part) + tr(Σ_part)) 
end


function getellipsepoints(cx, cy, rx, ry, θ)
    t = range(0, 2*pi, length=100)
    ellipse_x_r = @. rx * cos(t)
    ellipse_y_r = @. ry * sin(t)
    R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
    r_ellipse = [ellipse_x_r ellipse_y_r] * R
    x = @. cx + r_ellipse[:,1]
    y = @. cy + r_ellipse[:,2]
    (x, y)
end


function getellipsepoints(μ, Σ, confidence=0.95)
    quant = quantile(Chisq(2), confidence) |> sqrt
    cx = μ[1]
    cy = μ[2]
    
    egvs = eigvals(Σ)
    if egvs[1] > egvs[2]
        idxmax = 1
        largestegv = egvs[1]
        smallesttegv = egvs[2]
    else
        idxmax = 2
        largestegv = egvs[2]
        smallesttegv = egvs[1]
    end

    rx = quant*sqrt(largestegv)
    ry = quant*sqrt(smallesttegv)
    
    eigvecmax = eigvecs(Σ)[:,idxmax]
    θ = atan(eigvecmax[2]/eigvecmax[1])
    if θ < 0
        θ += 2*π
    end

    getellipsepoints(cx, cy, rx, ry, θ)
end

   
function plot_density_estimate(X, mc_samples, kwargs)
    dataname = kwargs["dataname"]
    method = kwargs["method"] 

    # Generate grid
    x_min, x_max = minimum(X[1, :]) - 1, maximum(X[1, :]) + 1
    y_min, y_max = minimum(X[2, :]) - 1, maximum(X[2, :]) + 1
    x_grid = range(x_min, x_max, length=100)
    y_grid = range(y_min, y_max, length=100) 
    xx = repeat(x_grid', length(y_grid), 1)
    yy = repeat(y_grid, 1, length(x_grid))
    grid_points = hcat(vec(xx), vec(yy))  

    # mc_samples = mc_samples[1:10]
    # Compute density for each grid point
    function compute_density(grid_point) 
        p = mc_samples |> length |> zeros
        Threads.@threads for i in eachindex(mc_samples)
            cnt = countmap(mc_samples[i].C) 
            pi = [cnt[j] for j in 1:length(unique(mc_samples[i].C))]
            pi = pi ./ sum(pi) 
            component_densities = [
                pi[k] * pdf(
                    MvNormal(
                        mc_samples[i].Mu[:, k], 
                        mc_samples[i].Sig[:, :, k]), 
                        grid_point) 
                for k in eachindex(pi)] 
            p[i] = sum(component_densities) 
        end 
        return mean(p)
    end

    density = zeros(size(grid_points, 1)) 
    for i in 1:size(grid_points, 1)
        density[i] = compute_density(grid_points[i, :]) 
    end
    density_matrix = reshape(density, (length(y_grid), length(x_grid)))
    println("Finish processing the density estimation computation")
 
    method = uppercase(method)

    # Plot
    # Plots.plotlyjs()
    # PlotlyKaleido.start()
    Plots.theme(:dao)
 
    logmeanexp(x) = logsumexp(x) - log(length(x))
    logcpo = round(
        - mean([
            let 
                x = X[:, i]
                pp = zeros(length(mc_samples))
                for (j, mc_sample) in enumerate(mc_samples) 
                    k = mc_sample.C[i]
                    G = MvNormal(mc_sample.Mu[:, k], mc_sample.Sig[:, :, k])
                    pp[j] = - logpdf(G, x)
                end     
                logmeanexp(pp)
            end  
            for i in 1:size(X, 2)]), 
        digits=3) 
    p = Plots.scatter(X[1, :], X[2, :], 
        framestyle=:grid, 
        markercolor=:grey,
        markerstrokewidth=0,
        leg=:best,
        alpha=0.7,  
        tickfontsize=10,
        xlabel=ifelse(dataname=="GvHD", "CD8", "x"), 
        ylabel=ifelse(dataname=="GvHD", "CD4", "y"),
        markersize=2, label="log-CPO: $(logcpo)", 
    )
    Plots.contour!(
        x_grid, y_grid, density_matrix, 
        cmap=:lajolla100,  
        levels=20, linewidth=0.5, alpha=0.8, cbar=false,
    )
 
    Plots.title!("Density Estimate by $(method)") 
    
    println("Finish plotting\n\n\n") 
    mkpath("./plots/$(dataname)")

    Plots.savefig(p, "./plots/$(dataname)/$(dataname)_$(method)_contour.pdf") 
end 


function plot_true_density_estimate(X, components, mixture_weights, kwargs)
    dataname = kwargs["dataname"]
    method = kwargs["method"] 

    # Generate grid
    x_min, x_max = minimum(X[1, :]) - 1, maximum(X[1, :]) + 1
    y_min, y_max = minimum(X[2, :]) - 1, maximum(X[2, :]) + 1
    x_grid = range(x_min, x_max, length=100)
    y_grid = range(y_min, y_max, length=100) 
    xx = repeat(x_grid', length(y_grid), 1)
    yy = repeat(y_grid, 1, length(x_grid))
    grid_points = hcat(vec(xx), vec(yy))  

    # mc_samples = mc_samples[1:10]
    # Compute density for each grid point
    function compute_density(grid_point)  
        p = similar(mixture_weights) .* 0
        component_densities = [
            mixture_weights[k] * pdf(components[k], grid_point) 
            for k in eachindex(components)] 
        p = sum(component_densities) 
        return p
    end

    density = zeros(size(grid_points, 1)) 
    Threads.@threads for i in 1:size(grid_points, 1)
        density[i] = compute_density(grid_points[i, :]) 
    end
    density_matrix = reshape(density, (length(y_grid), length(x_grid)))
    println("Finish processing the density estimation computation")
 
    method = uppercase(method)

    # Plot
    # Plots.plotlyjs()
    # PlotlyKaleido.start()
    Plots.theme(:dao)
 
    p = Plots.scatter(X[1, :], X[2, :], 
        framestyle=:grid, 
        markercolor=:grey,
        markerstrokewidth=0,
        leg=:best,
        alpha=0.7,  
        tickfontsize=10,
        xlabel=ifelse(dataname=="GvHD", "CD8", "x"), 
        ylabel=ifelse(dataname=="GvHD", "CD4", "y"),
        markersize=2, label=nothing #"log-CPO: $(logcpo)", 
    )
    Plots.contour!(
        x_grid, y_grid, density_matrix, 
        cmap=:lajolla100,  
        levels=20, linewidth=0.5, alpha=0.8, cbar=false,
    )
 
    Plots.title!("True Density") 
    
    println("Finish plotting\n\n\n") 
    mkpath("./plots/$(dataname)")

    Plots.savefig(p, "./plots/$(dataname)/$(dataname)_true_contour.pdf") 
end 


function plot_map_estimate(X, mc_samples, kwargs)  
    dataname = kwargs["dataname"]
    method = kwargs["method"] 

    map_est_ind = map(x -> x.lpost, mc_samples) |> argmax
    mc_sample = mc_samples[map_est_ind] 
  
    method = uppercase(method)
    # Plots.plotlyjs()
    # PlotlyKaleido.start()
    Plots.theme(:dao; palette=:tab20)

    p = Plots.scatter(X[1, :], X[2, :],   
        framestyle=:grid,
        alpha=1,
        markersize=2, 
        label=nothing, #"Log Posterior: $(round(mc_sample.lpost, digits=3))", 
        tickfontsize=10,
        xlabel=ifelse(dataname=="GvHD", "CD8", "x"), 
        ylabel=ifelse(dataname=="GvHD", "CD4", "y"),
        color=mc_sample.C)
    for k in unique(mc_sample.C) 
        Plots.plot!(
            getellipsepoints(       
                mc_sample.Mu[:, k], mc_sample.Sig[:, :, k], 0.95
            ),
            color=:black, label=nothing, linestyle=:dashdot, lw=1 
        )
    end 

    Plots.title!("MAP Component Estimate by $(method)") 
    
    println("Finish plotting\n\n\n") 
    mkpath("./plots/$(dataname)")
    Plots.savefig(p, "./plots/$(dataname)/$(dataname)_$(method)_map.pdf") 
end 

function plot_true_estimate(X, data_stats, kwargs)  
    dataname = kwargs["dataname"]
    method = kwargs["method"] 
    method = uppercase(method)
    
    # Plots.plotlyjs()
    # PlotlyKaleido.start()
    Plots.theme(:dao; palette=:tab20)

    p = Plots.scatter(X[1, :], X[2, :],   
        framestyle=:grid,
        alpha=1,
        markersize=2, 
        label=nothing, #"Log Posterior: $(round(mc_sample.lpost, digits=3))", 
        tickfontsize=10,
        xlabel=ifelse(dataname=="GvHD", "CD8", "x"), 
        ylabel=ifelse(dataname=="GvHD", "CD4", "y"),
        color=data_stats["C"])
    for component in data_stats["components"] 
        Plots.plot!(
            getellipsepoints(       
                component.μ, component.Σ, 0.95
            ),
            color=:black, label=nothing, linestyle=:dashdot, lw=1 
        )
    end 

    n = split(dataname, "_")[end]
    Plots.title!("True Component Assignment of Simulation $n") 
    
    println("Finish plotting\n\n\n") 
    mkpath("./plots/$(dataname)")
    Plots.savefig(p, "./plots/$(dataname)/$(dataname)_true_component.pdf") 
end 


function plot_min_d_all(X, mc_sample_dict, kwargs)
    dist_type = kwargs["dist_type"] 
    
    function compute(mc_samples)
        min_d_vec = zeros(length(mc_samples))
        for (k, mc_sample) in enumerate(mc_samples)
            K = size(mc_sample.Mu, 2)  
            d_mat = fill(Inf, K, K)
            indices = filter(c -> c[1] < c[2], CartesianIndices((1:K, 1:K)))
            Threads.@threads for (i, j) in Tuple.(indices) 
                if dist_type == "Mean"
                    d = (mc_sample.Mu[:, i] .- mc_sample.Mu[:, j]) .^2 |> sum |> sqrt
                elseif dist_type == "Wasserstein"  
                    d = wass_dist(
                        mc_sample.Mu[:, i], mc_sample.Sig[:, :, i],
                        mc_sample.Mu[:, j], mc_sample.Sig[:, :, j])
                else 
                    throw("Distance type not supported")
                end 
                d_mat[i, j] = d 
            end 
            min_d_vec[k] = minimum(d_mat)  
        end
        min_d_vec 
    end 

    Plots.plotlyjs() 
    PlotlyKaleido.start()
    Plots.theme(:dao)

    p = nothing  
    dfs = []
    for (i, (method, mc_samples)) in enumerate(mc_sample_dict)  
        method = uppercase(method)
        df = DataFrame(x=compute(mc_samples), method=method)  
        push!(dfs, df)    
    end 
    df = vcat(dfs...)
 
    p = @df df density(:x, group=:method, # label=method, 
        tickfontsize=10, lw=3, linestyle=:auto,  
        title="Density of Minimal Inter-Component $(dist_type) Distance")
 
    dataname = kwargs["dataname"]
    println("Finish plotting\n\n\n") 

    mkpath("./plots/$(dataname)")
    Plots.savefig("./plots/$(dataname)/$(dataname)_$(dist_type)_min_dist_kde.svg")
end 


function plot_min_d(X, mc_samples, kwargs)
    dataname = kwargs["dataname"]
    method = kwargs["method"]  

    min_d_vec = zeros(length(mc_samples))
    for (k, mc_sample) in enumerate(mc_samples)
        K = size(mc_sample.Mu, 2)  
        d_mat = fill(Inf, K, K)
        indices = filter(c -> c[1] < c[2], CartesianIndices((1:K, 1:K)))
        Threads.@threads for (i, j) in Tuple.(indices) 
            d = (mc_sample.Mu[:, i] .- mc_sample.Mu[:, j]) .^2 |> sum |> sqrt
            d_mat[i, j] = d 
        end 
        min_d_vec[k] = minimum(d_mat)  
    end 

    # Plot 
    p = plot(x=min_d_vec, Theme(alphas=[0.6]),
        Stat.density, 
        Guide.title("KDE of minimal mean distance with $(method) Repulsion")) 
 
    mkpath("./plots/$(dataname)")
    draw(PDF("./plots/$(dataname)/$(dataname)_min_dist.svg", 4inch, 3inch), p)
    println("Finish plotting\n\n\n")
end 


function load_and_plot(kwargs)
    display(kwargs)  

    dataname = kwargs["dataname"]
    method = kwargs["method"]
    X = load_data(dataname) 
    
    mkpath("./plots/") 
         
    if method == "all"
        methods = occursin("sim", dataname) ?
            ["mfm-full", "rgm-full", "wrgm-full"] :
            ["mfm-full", "rgm-full", "wrgm-full", "mfm-diag", "rgm-diag", "wrgm-diag"] 
        mc_sample_dict = OrderedDict( 
            JLD2.load("results/$(dataname)_$(method).jld2", "mc_samples")
            for method in methods
        )
        plot_min_d_all(X, mc_sample_dict, kwargs) 
    else
        if method != "true" 
            mc_samples = JLD2.load(
                "results/$(dataname)_$(method).jld2", "mc_samples") 
            plot_density_estimate(X, mc_samples, kwargs)
            plot_map_estimate(X, mc_samples, kwargs)
        else 
            data_stats = JLD2.load("data/$dataname.jld2") 
            plot_true_density_estimate(
                X, data_stats["components"], data_stats["mixture_weights"], kwargs)
            plot_true_estimate(X, data_stats, kwargs)
        end 
    end   
end 


load_and_plot(parse_plot_cmd())
