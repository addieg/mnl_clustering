## Title: Complete ratio matrices for different K to perform hypothesis testing.
## Author: Andreea Georgescu
## Date: 05-21-2019

using Distributions, JuMP, Ipopt, DataFrames, CSV, StatsBase
module cl
include("functions.jl")
end

num = parse(Int64,ARGS[1])

function get_stats(mixtures, ratio_data, sales_data, assorts_data)
    complete_ratios = @timed cl.complete_matrix(ratio_data, mixtures, true)

    model_found = convert(DataFrame, complete_ratios[1][3])
    model_found[:probability] = [1/mixtures for i in 1:mixtures]

    return( [complete_ratios[2], complete_ratios[1][2], cl.logLikelihood(model_found, sales_data, assorts_data)])
end
    
function run_hyp_test(num)
    sales = readtable("../data/sales_dataset$num.csv")
    assorts = readtable("../data/assorts_dataset$num.csv")
    ratios = cl.get_ratio_data(sales) 

    stats = map((x) -> get_stats(x, ratios, sales, assorts), 1:20)

    final_stats = zeros(20,3)
    for i in 1:20
        final_stats[i,:] = stats[i]
    end
    return(convert(DataFrame, final_stats)) 
end

result = run_hyp_test(num)
CSV.write("../hyp_test/results_dataset$num.csv", result)