## Title: Functions for clustering kMNL
## Author: Andreea Georgescu
## Date: 05-20-2019
## Last update: 05-23-2019


using Distributions, JuMP, Ipopt, DataFrames, CSV, StatsBase, Plots


## function returns a pmf with support 1:k
## such that no probability share is less than 1/k^2.
function random_pmf(k)
    pmf = zeros(k)
    remaining_shares = k^2
    for i in 1:(k-1)
        shares = rand(1:(remaining_shares-(k-i)))
        pmf[i] = float(shares)/(k^2)
        
        remaining_shares -= shares
    end
    pmf[k] = float(remaining_shares)/(k^2)
    return(shuffle(pmf))
end

## function takes input: 
## <products> the number of products
## <models> the number of models mixed
## <no_purchase_max> true if no-purchase has greatest location parameter; if not parameters are in [0.5,5]
## funciton returns the parameters of a mixed MNL where all MNLs mixed have the same (permuted) utility vector
function generate_model(products, models, no_purchase_max)   
    if no_purchase_max
        lambda = rand(Uniform(0.1,1),products)
    else
        lambda = rand(Uniform(0.5,5),products)
    end

    model = zeros(models, products)
    for m in 1:models
        model[m,:] = shuffle(lambda)
    end
    model = convert(DataFrame, model)
    model[:probability] =  random_pmf(models)
    return(model)
end 

# function returning assortment, given parameters items (# of products possible),
# and assortment_size (range of sizes acceptable for assortemnts).
function new_assortment(items, assortment_size)
    assort = [zeros(items);[1]]
    size = rand(Int(floor(assortment_size[1]*items)):Int(floor(assortment_size[2]*items)))
    indices = shuffle(1:items)[1:size]
    
    assort[indices] = ones(size)
    return(assort)
end

## function takes input:
## <model> the parameters of a mixed MNL model
## <assortments> the number of assortments offered
## <observations> the number of observations per assortment
## <assortment_size> assortments size
function generate_data(model, datapoints, observations, assortment_size)
    products = ncol(model)-1
    
    # create data to generate
    assorts = zeros(datapoints, products+1)
    sales = zeros(datapoints, products+1)   
    
    for n in 1:datapoints
        assorts[n,:] = new_assortment(products, assortment_size)
    end
    
    for n in 1:datapoints
        shares = [assorts[n,i]*sum(model[:probability][m] * model[m,i]/
                ( 1+ sum( assorts[n,j] * model[m,j] for j in 1:products)) for m in 1:nrow(model)) for i in 1:products]     
        
        shares = [shares ; [1-sum(shares)]]          
        for i in 1:observations
            assign = Int(rand(Categorical(shares)))
            sales[n,assign] += 1
        end
    end
    
    return(model, convert(DataFrame, sales), convert(DataFrame, assorts))
end

## function takes input:
## <sales_data> dataframe with sales observations
## returns the ratio dataframe associated.
function get_ratio_data(sales_data)   
    ratio_data = sales_data[sales_data[:,ncol(sales_data)] .> 0,:]
    for i in 1:ncol(ratio_data)
        ratio_data[:,i] = ratio_data[:,i] ./ ratio_data[:,ncol(ratio_data)]
    end
    return(ratio_data[:,1:(ncol(ratio_data)-1)])
end


## function taking input:
## <sales_data> dataframe with sales data
## <assortment_data> df with assortment data
## returns MLE MNL.
function fit_mnl(sales_data, assortment_data)
    products = ncol(sales_data)
    
    model = Model(solver=IpoptSolver(print_level=1))
    @variable(model, lam[i=1:products] >= 0.000001)
    @constraint(model, lam[products] == 1)
    
    @NLexpression(model, denom[n=1:nrow(sales_data)], sum(  assortment_data[n,j]*lam[j] for j in 1:products) )
    @NLexpression(model, logll[n=1:nrow(sales_data)], sum(  sales_data[n,j]*log(lam[j]/denom[n])  for j in 1:products))

    @NLobjective(model, Max, sum( logll[n]  for n in 1:nrow(sales_data)))
    solve(model)

    return(getvalue(lam)[1:(length(getvalue(lam))-1)], -getobjectivevalue(model))
end


function fit_mnl_ratios(ratio_data, assortment_data, zero_observations)
    products = ncol(ratio_data)
    
    model = Model(solver=IpoptSolver(print_level=1))
    @variable(model, lam[i=1:products] >= 0.000001)
    
    R = [1 + sum(  assortment_data[n,j]*ratio_data[n,j] for j in 1:products) for n in 1:nrow(ratio_data)]
    
    @NLexpression(model, L[n=1:nrow(ratio_data)], 
        sum(assortment_data[n,j]*ratio_data[n,j]*log(lam[j]) for j in 1:products) - R[n]*log(1 + sum(  assortment_data[n,j]*lam[j] for j in 1:products)) )
    
    @NLobjective(model, Max, sum( zero_observations[n,1] * L[n]  for n in 1:nrow(ratio_data)))
    solve(model)
    
    return(getvalue(lam), -getobjectivevalue(model))
end


## function taking input
## <model> a mixed MNL model
## <assortment_datum> one assortment, 0-1 indexed, dataframe with one row.
## returns purchase probabilities.
function compute_shares(model, assortment_datum)
    products = ncol(model) - 1
    shares = [assortment_datum[1,i] * sum(model[:probability][m] * model[m,i] /
            (1 + sum(assortment_datum[1,j] * model[m,j] for j in 1:products)) for m in 1:nrow(model)) for i in 1:products]
    shares = [shares; [1-sum(shares)]]
    
    return(shares)
end


## function taking input
## <model> a mixed MNL model
## <sales_data> each row has the number of sales for each product, given some assortment
## <assortment_data> each row has the assortment corresponding to sales, 0-1 indexed
## returns negative log likelihood

function logLikelihood(model, sales_data, assortment_data)
    products = ncol(model) 
    
    ll = 0
    for n in 1:nrow(sales_data)
        shares = compute_shares(model, assortment_data[n,:])
        ll += sum(sales_data[n,i]*log(shares[i])  for i in 1:products if assortment_data[n,i] > 0)
        
    end
    return(-ll)
end 
            
## function taking input
## <df> a dataframe with entries ratios of probabilities, and many entries zero.
## <K> the number of mixtures we assume are in the underlying model
## returns a complete matrix, replacing zero entries with appropriate linear combinations.
function complete_matrix(df, K, no_purchase_max)
    model = Model(solver=IpoptSolver(print_level=1))
    @variable(model, 0.0001 <= weights[n in 1:nrow(df), k in 1:K] <= 1)
                
                
    if no_purchase_max
        @variable(model, 0 <= lams[k in 1:K, i in 1:ncol(df)] <= 1)
    else
        @variable(model, lams[k in 1:K, i in 1:ncol(df)] >= 0)
    end 
                
    @variable(model, s[n in 1:nrow(df), i in 1:ncol(df)] >= 0)
    @variable(model, sl[n in 1:nrow(df), i in 1:ncol(df)] >= 0)
    
    for n in 1:nrow(df)
        for i in 1:ncol(df) if df[n,i] > 0
                @NLconstraint(model, df[n,i] + s[n,i] == sl[n,i] + sum(weights[n,k] * lams[k,i] for k in 1:K))
            end
        end
    end
    
    @NLobjective(model, Min, sum(s[n,i] + sl[n,i] for n in 1:nrow(df), i in 1:ncol(df)))
    solve(model)
    
    w = getvalue(weights)
    l = getvalue(lams)
    
    for n in 1:nrow(df)
        for i in 1:ncol(df) if df[n,i] == 0
                df[n,i] = sum(w[n,k] * l[k,i] for k in 1:K)
            end
        end
    end
    
    return(df, getobjectivevalue(model), l)
end
            
## function taking input
## <p> and <q> df rows with ratios of probabilities
## returns distance between p and q, i.e. log likleihood of data with ratios p under model with lambda q.
function distance(p,q)
    new_p = [p[1,i] for i in 1:ncol(p) if p[1,i] * q[1,i] > 0]
    new_q = [q[1,i] for i in 1:ncol(p) if p[1,i] * q[1,i] > 0]
    
    
    if length(new_p) == 0
        return(0)
    end
    
    #dist = sum(new_p .* log.(new_q)) - (1 + sum(new_p))*log(1+sum(new_q))
    dist = sum(new_p .* log.(new_q ./ new_p)) - (1 + sum(new_p))*log((1+sum(new_q))/(1+sum(new_p)))
                                        
    #return(-dist / length(new_p)  )
    return(-dist)                                    
end
                                    
                                    
## function taking input
## <sales_data> each row has an observation, i.e. number of sales per product
## <assortment_data> each row has the assortment corresponding to sales, 0-1 indexed
## <K> number of mixed models
## returning a mixed MNL model by k-means ++ 
function initialize_model_plusplus(ratios_data, K)
    products = ncol(ratios_data)
    model = zeros(K,products)
    
    # pick clever start
    cluster_centers = zeros(K)
    cluster_centers[1] = Int(rand(1:nrow(ratios_data)))

    for k in 2:K
        dist = zeros(nrow(ratios_data))

        for n in 1:nrow(ratios_data)
            dist[n] = 
            minimum([ distance(ratios_data[n,:], ratios_data[Int(cluster_centers[j]),:])   for j in 1:(k-1)])
        end

        seed_prob = dist ./ sum(dist)
        cluster_centers[k] = Int(rand(Categorical(seed_prob)))
    end

    #print(cluster_centers)
    
    for k in 1:K
        model[k,:] = convert(Array, ratios_data[Int(cluster_centers[k]),:])
    end
    model = convert(DataFrame, model)
    model[:probability] = [1.0/K for k in 1:K]

    return(model)
end 

## function taking input
## <sales_data> each row has an observation, i.e. number of sales per product.
## <K> the number of clusters to create
## returning list of K datasets, by assigning observations to dataset randomly.
function random_assign(sales_data, K)
    assigned_sales = [convert(DataFrame, zeros(nrow(sales_data), ncol(sales_data))) for k in 1:K]
    
    for n in 1:nrow(sales_data)
        for i in 1:ncol(sales_data)
            
            new_sales = zeros(K)
            
            for j in 1:sales_data[n,i]
                assign = Int(rand(Categorical((1.0/K) .* ones(K))))
                new_sales[assign] += 1
            end
            
            for k in 1:K
                assigned_sales[k][n,i] = new_sales[k]
            end
            
        end
    end
        
    
    return(assigned_sales)
end
                                    
                                    
## function taking input
## <sales_data> each row has an observation, i.e. number of sales per product
## <assortment_data> each row has the assortment corresponding to sales, 0-1 indexed
## <K> number of mixed models
## returning a mixed MNL model by splitting observations randomly in K clusters
function initialize_model_mnl(sales_data, assortment_data,  K)
    products = ncol(sales_data) - 1 
    model = zeros(K,products)
    
    # create random clusters
    assigned_sales = random_assign(sales_data, K)

    for k in 1:K
        model[k,:] = fit_mnl( assigned_sales[k], assortment_data)[1]
    end
    model = convert(DataFrame, model)
    model[:probability] = [sum(convert(Array, assigned_sales[k])) / sum(convert(Array, sales_data)) for k in 1:K]

    return(model)
end 
                                    
                                    
                                    
## function taking input
## <sales_data> each row has an observation, i.e. number of sales per product
## <assortment_data> each row has the assortment corresponding to sales, 0-1 indexed
## <K> number of mixed models
## returning a mixed MNL model by choosing randomly K observations.
function initialize_model_random(sales_data, assortment_data, K)
    products = ncol(assortment_data) - 1
    model = zeros(K,products)
    
    a = shuffle(1:nrow(sales_data))
    cluster_centers = [a[i] for i in 1:K]
    #print(cluster_centers)
    
    for k in 1:K
        model[k,:] = fit_mnl(sales_data[Int(cluster_centers[k]),:], assortment_data[Int(cluster_centers[k]),:])[1]
    end
    model = convert(DataFrame, model)
    model[:probability] = [1.0/K for k in 1:K]
        
    return(model)
end 
                                    
                                    
                                    
## function taking input
## <assortment_data> each row has the assortment corresponding to sales, 0-1 indexed
## <model> a mixed MNL model
## <k> an integer indicating one of the mixed models
## returns an array, same size as assortment_data, where entry (n,i) is the probability that observation of sale of i
#### in assortment n, was generated by model k
function compute_posterior(assortment_data, model, k)
    products = ncol(assortment_data) - 1
    posterior = zeros(nrow(assortment_data), ncol(assortment_data))
    
    for n in 1:nrow(assortment_data)
        shares = compute_shares(model, assortment_data[n,:])
        shares_k = [assortment_data[n,i] * model[k,i] /
                (1 + sum(assortment_data[n,j] * model[k,j]  for j in 1:products) ) for i in 1:products]
        shares_k = [shares_k; [1-sum(shares_k)]]
        
        for i in 1:ncol(assortment_data)
            if assortment_data[n,i] > 0
                posterior[n,i] = shares_k[i]*model[:probability][k] / shares[i]
            else
                posterior[n,i] = 0
            end
        end
    end
    return(posterior)
end
                                    
                                    
## function taking input
## <sales_data> each row has an observation, i.e. number of sales per product
## <assortment_data> each row has the assortment corresponding to sales, 0-1 indexed
## <initial_model> a mixed MNL model 
## <iterations> iterations limit
## runs EM on the sales_data, starting at initial_model, until convergence or iteration elapsed.
function run_mmnl_em(sales_data, assortment_data, initial_model, iterations)
    products = ncol(assortment_data) - 1
    K = nrow(initial_model)
    
    # create likelihoods string 
    likelihoods = zeros(iterations+1)
    likelihoods[1] = logLikelihood(initial_model, sales_data, assortment_data)
    
    
    # iterate
    centroids = zeros(K,products)
    lambda = zeros(K)

    for t in 1:iterations
        # apply EM iteration
        for k in 1:K
            # E-step
            post = compute_posterior(assortment_data, initial_model, k)
            sales = convert(DataFrame, convert(Array, sales_data) .* post )
            
            # M-step
            centroids[k,:] = fit_mnl(sales, assortment_data)[1]
            lambda[k] = sum(convert(Array, sales)) / sum(convert(Array, sales_data))
        end
        
        # update model
        for k in 1:K
            for i in 1:products
                initial_model[k,i] = centroids[k,i]
            end
            initial_model[k,products+1] = lambda[k]
        end
        
        # append likelihood
        likelihoods[t+1] = logLikelihood(initial_model, sales_data, assortment_data)
        
    end 
    
    # return the model found, and the likelihoods
    return(likelihoods)
end

## function taking input
## <lls> a dataframe with each row a log likelihood path taken by EM (normalized by true likelihood, i.e. difference);
## --- also has a column :method with entries plus / random / mnl. and all other columns are at the end.
## <first_iteration> first iteration to plot
## <last_iteration> last iteration to plot
## <y_range> a positive number, the limit of the y axis range
## <num> number of the model, to be printed in plot title  
## <extra_method> if there is a fourth method in the df lls (Bool)
function plot_paths(lls, first_iteration, last_iteration, y_range, num, extra_method)
    # plotting settings
    gr()
    gr(fmt=:png)
                                        
    stats_plus = lls[lls[:method] .== "plus", :]
    stats_random = lls[lls[:method] .== "random", :]
    stats_mnl = lls[lls[:method] .== "mnl", :]

    alph = 0.3

    # plot mnl lines
    plot([i for i in first_iteration:last_iteration], reshape(convert(Array, stats_mnl[1, first_iteration:last_iteration]), last_iteration-first_iteration+1), color = "purple", linealpha = alph, label = "")
    for k in 2:nrow(stats_random)
        plot!([i for i in first_iteration:last_iteration], reshape(convert(Array, stats_mnl[k, first_iteration:last_iteration]), last_iteration-first_iteration+1), color = "purple", linealpha = alph, label = "")
    end
    mean_mnl = [mean(stats_mnl[k,i] for k in 1:nrow(stats_mnl)) for i in 1:last_iteration]
    plot!([i for i in first_iteration:last_iteration], mean_mnl[first_iteration:last_iteration], color = "purple", linewidth = 3, label = "Random Assign")


    # plot plus lines
    for k in 1:nrow(stats_plus)
        plot!([i for i in first_iteration:last_iteration], reshape(convert(Array, stats_plus[k, first_iteration:last_iteration]), last_iteration-first_iteration+1), color = "green", linealpha = alph, label = "")
    end

    # plot random lines
    for k in 1:nrow(stats_random)
        plot!([i for i in first_iteration:last_iteration], reshape(convert(Array, stats_random[k, first_iteration:last_iteration]), last_iteration-first_iteration+1), color = "orange", linealpha = alph, label = "")
    end


    mean_plus = [mean(stats_plus[k,i] for k in 1:nrow(stats_plus)) for i in 1:last_iteration]
    plot!([i for i in first_iteration:last_iteration], mean_plus[first_iteration:last_iteration], color = "green", linewidth = 3, label = "Plus Plus")

    if extra_method
        stats_extra = lls[(lls[:method] .!= "mnl") .& (lls[:method] .!= "plus") .& (lls[:method] .!= "random"), :] 
        
        # plot extra lines
        for k in 1:nrow(stats_extra)
            plot!([i for i in first_iteration:last_iteration], reshape(convert(Array, stats_extra[k, first_iteration:last_iteration]), last_iteration-first_iteration+1), color = "light green", linealpha = alph, label = "")
        end

        mean_extra = [mean(stats_extra[k,i] for k in 1:nrow(stats_extra)) for i in 1:last_iteration]
        plot!([i for i in first_iteration:last_iteration], mean_extra[first_iteration:last_iteration], color = "light green", linewidth = 3, label = "Extra")
    end
    
    mean_random = [mean(stats_random[k,i] for k in 1:nrow(stats_random)) for i in 1:last_iteration]
    plot!([i for i in first_iteration:last_iteration], mean_random[first_iteration:last_iteration], color = "orange", linewidth = 3, 
        label = "Random Datapoints", ylabel = "Difference in log-likelihood", 
        title = "Difference in log-likelihood from true model, in model $num", yaxis = [0,y_range])
                                        
end