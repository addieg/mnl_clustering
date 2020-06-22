## Title: Initialize with different starts and run 5-step EM
## Author: Andreea Georgescu
## Date: 05-21-2019
## Last update: 05-23-2019

using Distributions, JuMP, Ipopt, DataFrames, CSV, StatsBase
module cl
include("functions.jl")
end

num = parse(Int64,ARGS[1])



function run_diff_starts(true_model, sales_data, assortment_data, iterations, samples )
    # compute full ratio matrix
    mixtures = nrow(true_model)
    
    # initialize MNL
    mnl_starts = [cl.initialize_model_mnl(sales_data, assortment_data,  mixtures) for i in 1:samples]
    
    # initialize random
    random_starts = [cl.initialize_model_random(sales_data, assortment_data,  mixtures) for i in 1:samples]
    
    # initialize plus plus
    ratios = cl.get_ratio_data(sales_data)
    ratio_data = cl.complete_matrix(ratios, mixtures, false)
    plus_starts = [cl.initialize_model_plusplus(ratio_data[1], mixtures) for i in 1:samples]
    
    # initialize plus model with extra knowledge
    ratios = cl.get_ratio_data(sales_data)
    extra_ratio_data = complete_matrix(ratios, mixtures, true_model)
    extra_plus_starts = [cl.initialize_model_plusplus(extra_ratio_data[1], mixtures) for i in 1:samples]
    
    likes = zeros(4*samples,iterations+1)
    
    for i in 1:samples
        likes[i,:] = cl.run_mmnl_em(sales_data, assortment_data, mnl_starts[i], iterations)
    end

    for i in 1:samples
        likes[i + samples,:] = cl.run_mmnl_em(sales_data, assortment_data, random_starts[i], iterations)
    end
    
    for i in 1:samples
        likes[i + 2*samples,:] = cl.run_mmnl_em(sales_data, assortment_data, plus_starts[i], iterations)
    end
    
    for i in 1:samples
        likes[i + 3*samples,:] = cl.run_mmnl_em(sales_data, assortment_data, extra_plus_starts[i], iterations)
    end
    
    likes = convert(DataFrame, likes)
    likes[:method] = [repeat(["mnl"],samples); repeat(["random"],samples); repeat(["plus"],samples); repeat(["extra"],samples)]
    true_ll = cl.logLikelihood(true_model, sales_data, assortment_data)
    likes[:true_ll] = true_ll .* ones(4*samples)
    
    return(likes)
end


### DO THE WORK ###

sales = CSV.read("./data/sales$num.csv")
assorts = CSV.read("./data/assorts$num.csv") 
model = CSV.read("./data/model$num.csv")

result = run_diff_starts(model, sales, assorts, 30, 20 )
CSV.write("./diff_starts/em_results$num.csv", result)

    