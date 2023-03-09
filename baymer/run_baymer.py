#!/usr/bin/Python

# Created by: Christopher J Adams 8/30/2022
# 

###############################################################################
###
### This script will run the metropolis hastings mcmc where multiplications
### are made down the tree. All parameters are estimately jointly and phis
### come from a mixture of normals and are sampled in log space. 
###
###############################################################################


## NOTE Assumes symmetric-ish counts

#import cProfile
import sys
import getopt
import os
import yaml
import json
import time
import numpy as np
import random
from itertools import product
import numba
from numba import njit, prange, vectorize, float64

def help(exit_num=1):
    print("""-----------------------------------------------------------------
ARGUMENTS
    -c => <yaml> count json config file REQUIRED
    -p => <yaml> parameter values config file REQUIRED
    -r => <int> Index of random seed in parameter yaml OPTIONAL Default: 0           
    -z => <bool> initialize starting phis to spike and indicator = 1 OPTIONAL
    -t => <int> set number of threads to use OPTIONAL Defaults to the max available
""")
    sys.exit(exit_num)

## GLOBAL VARS
BATCH = 50

###############################################################################
###########################  COLLECT AND CHECK ARGUMENTS  #####################
###############################################################################

## MAIN ##
# gather all arguments

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "c:p:r:zat:")
    except getopt.GetoptError:
        print("Error: Incorrect usage of getopts flags!")
        help()

    options_dict = dict(opts)

    ## Required arguments
    try:
        data_config_file = options_dict['-c']
        param_config_file = options_dict['-p']
        
    except KeyError:
        print("Error: One of your required arguments does not exist.")
        help()

    # Optional arguments
    random_seed_index = int(options_dict.get('-r', 0))
    num_threads = int(options_dict.get('-t', False))
    zero_init = options_dict.get("-z", False)
    if zero_init == "":
        zero_init = True


    print("Acceptable Inputs Given")
    
    driver(data_config_file, param_config_file, random_seed_index, zero_init, num_threads)


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(data_config_file, param_config_file, random_seed_index,  zero_init = False, num_threads = False):
    
    if num_threads:
        numba.set_num_threads(num_threads)

    print("Number of threads: ", numba.get_num_threads())
    
    suppress = True
    #suppress = False
    
    # load config files that specify model parameters and file locations 
    config_dict = yaml.load(open(data_config_file, 'r'), Loader=yaml.SafeLoader)
    param_config_dict = yaml.load(open(param_config_file, 'r'), Loader=yaml.SafeLoader)

    # set random seeds
    random_seed = param_config_dict["random_seeds"][random_seed_index]
    random.seed(int(random_seed))
    np.random.seed(int(random_seed))

    ## init hyperparameters from config
    c = param_config_dict['c']
    max_mer = param_config_dict['max_mer']
    output_dir = param_config_dict["posterior_dir"]
    dataset = param_config_dict["dataset"]
    pop = param_config_dict['pop']

    oppo_asymmetry = False
    alternation_pattern = param_config_dict['alternation_pattern']
    if alternation_pattern == "left":
        oppo_asymmetry = True
    
    feature = param_config_dict['feature']
    num_iterations, burnin = param_config_dict[0]['iteration_burnin']
    
    # collect the necessary leaf count data
    leaf_count_dict = prep_leaf_count_dict(config_dict, pop, feature, max_mer, dataset)
    
    # init the context array
    context_list = ['phi_naught']
    # init the p_vec array
    p_vec_array = None
    sample_order = list(range(num_iterations))
    first_sample = 0
    # for each edge, holds the set p vector for the edge's parent in the previous layer
    set_p_vec_sample_array = np.array([])
    set_p_vec_array = np.array([0.018, 0.011, 0.014])
    # for each edge, holds the set probabilities for the previous layer
    data_layer = -1
    
    for layer in range(data_layer + 1, max_mer):
        print('================================\nTree layer: {}\n================================'.format(layer))
        
        layer_size = (4 ** layer) * 2
        total_leaves = 4**(max_mer - layer)
        print("layer_size: ", layer_size)
        print("total_leaves: ", total_leaves)
        
        
        # collect proposal sigmas
        sigma_sigma = np.float32(param_config_dict[layer]['proposal_sigmas']['sigma_sigma'])
        proposal_sigma = np.float32(param_config_dict[layer]['proposal_sigmas']['phi_sigma'])
        slab_sigma = np.float32(param_config_dict[layer]['slab_sigma'])

        indicator_sampling = "gibbs"
        
        set_sigma = param_config_dict["set_sigma"]
        if set_sigma:
            slab_sigma = np.float32(set_sigma)
        
        # collect starting values
        alpha = param_config_dict[layer]['alpha']
        
        spike_sigma = np.float32(slab_sigma * c)
        sigmas = (spike_sigma, slab_sigma)
        alpha_sigma = param_config_dict[layer]['proposal_sigmas']['alpha_sigma']
        
        spike_alpha_prob = np.log(1-alpha)
        slab_alpha_prob = np.log(alpha)
        alpha_probs = (spike_alpha_prob, slab_alpha_prob)
        
      

        starting_hyperparameters = (proposal_sigma, sigma_sigma, c, slab_sigma, alpha, alpha_sigma, indicator_sampling)
        ## set starting conditions for next layer based on the randomly sampled order
        set_probability_vector = 0
        set_p_vec_sample_array = np.array([])
        if layer > 0:

            first_sample = np.random.choice(post_thin_samples)
            parent_layer_set_p_vec_sample_array = set_p_vec_sample_matrix[first_sample]
            set_p_vec_array = project_set_p_vec_sample_array(parent_layer_set_p_vec_sample_array, layer_size)
            parent_layer_set_probability_vector = old_set_probabilities_matrix[first_sample]
            #print(parent_layer_set_probability_vector.shape)
            #print(parent_layer_set_probability_vector[0:10])
            set_probability_vector = project_set_p_vec_sample_array(parent_layer_set_probability_vector, layer_size) * 0.25
            #print(set_probability_vector.shape)
            #print(set_probability_vector[0:10])
            #print(np.sum(set_probability_vector))

        num_iterations, burnin = param_config_dict[layer]['iteration_burnin']
        
        # get the arrays for each edge in this layer
        phi_array, p_vec_array, indicator_array, context_list, leaf_counts_array, phi_probabilities_array, alpha_probabilities_array, zero_contexts = initialize_layer_tree_edge_list(layer, layer_size, max_mer, context_list, sigmas, alpha_probs, set_p_vec_array, leaf_count_dict, first_sample, random_seed, zero_init, oppo_asymmetry)
        print("layer arrays initialized")

        leaf_likelihood_array = get_leaf_likelihood_array(layer_size, leaf_counts_array, p_vec_array, zero_contexts) 
        
        ## init numpy data sample matrices
        thinning_interval = param_config_dict[layer]['thinning_parameter']
        total_samples = int((num_iterations - burnin) / thinning_interval)

        total_batches = int(num_iterations / BATCH)


        likelihood_matrix = np.zeros((num_iterations + 1, 5))
        phi_sample_matrix = np.ones((total_samples, layer_size, 3), dtype = np.float32)
        rate_matrix = np.zeros((total_samples, layer_size, 3), dtype = np.float32)
        indicator_sample_matrix = np.zeros((total_samples, layer_size, 3), dtype = np.int32)
        alpha_sample_matrix = np.zeros(total_samples, dtype = np.float32)
        #alpha_probability_sample_matrix = np.zeros((total_batches, layer_size * 3), dtype = np.float32)
        
        set_probabilities_matrix = np.zeros((total_samples, layer_size, 3), dtype = np.float32)

        phi_sigma_sample_matrix = np.zeros((total_batches, layer_size * 3), dtype = np.float32)
        phi_alpha_sample_matrix = np.zeros((total_batches, layer_size * 3), dtype = np.float32)
        phi_acceptance_batch_array = np.zeros((total_batches, layer_size * 3), dtype = np.float32)
  
        phi_sigma_array = np.full(layer_size * 3, proposal_sigma, dtype = np.float32)
        
        phi_batch_array = np.zeros(layer_size * 3, dtype = np.float32)
        
        sigma_batch_array = np.array([0.0])
        phi_sampling_alpha_array = np.full(layer_size * 3, 0.5, dtype = np.float32)
        phi_indicator_count_array = np.zeros(layer_size * 3, dtype = np.int32)
        sigma_sample_matrix = None
        if not set_sigma:
            sigma_sample_matrix = np.zeros(total_samples, dtype = np.float32)
        
        # calculate the probabilities/likelihoods for each parameter
        total_phi_probability = np.sum(phi_probabilities_array)
        total_alpha_probability = np.sum(alpha_probabilities_array)
        total_leaf_likelihood = np.sum(leaf_likelihood_array)
        total_set_probability = np.sum(set_probability_vector)
        total_posterior = total_phi_probability + total_alpha_probability + total_set_probability + total_leaf_likelihood
        
        print("Total posterior from phi: ", total_phi_probability)
        print("Total posterior from alpha: ", total_alpha_probability)
        print("Total posterior from likelihood: ", total_leaf_likelihood)
        print("Total posterior from set probability: ", total_set_probability)
        print("JOINT POSTERIOR: ", total_posterior)
        
        likelihood_matrix[0] = [total_phi_probability, total_alpha_probability, total_leaf_likelihood, total_set_probability, total_posterior]
        
        iteration = 0
        
        ## timing
        timing = True
        
        total_it = 0
        it_time_start = 0
        sub_it_time_start = 0

        sigma_accepted_array = np.array([0])
        alpha_accepted_array = np.array([0])
        phi_accepted_array = np.array([0])
        
        thinned_iteration = 0
        post_burnin = False
        adaptive_winddown = False
        #post_burnin_test = False
        batch = False
        ## loop through all the sample iterations for this layer
        while iteration < num_iterations:
            
            #if iteration % 1000 == 0:
            #    print("iteration ", iteration)
            
            iteration += 1
            if not post_burnin:
                if not adaptive_winddown:
                    adaptive_winddown_window = min([20000, int(burnin / 2)])
                    if iteration > (burnin - adaptive_winddown_window):
                        last_x_batches = min([50, int(adaptive_winddown_window / BATCH)])
                        adaptive_winddown = True
                        ## set the proposal parameters based on the average of the previous 500
                        batch = int(iteration / BATCH)
                        phi_sigma_array = np.mean(phi_sigma_sample_matrix[batch-last_x_batches:batch, :], axis = 0)
                        phi_sampling_alpha_array = np.mean(phi_alpha_sample_matrix[batch-last_x_batches:batch, :], axis = 0)
                elif iteration > burnin:
                    post_burnin = True
            
            if iteration % BATCH == 0:
                batch = int(iteration / BATCH)
            
            if timing and iteration == 2:
                it_time_start = time.time()
             
            if not suppress:
                print('---------------------------------------------------------------------------------------------')
                print("Iteration: ", iteration, " Layer: ", layer)
                sub_it_time_start = time.time()
            start = 0
            
            if layer > 0:
                               
                #### Using the current sample, initialize the state of the layers of the tree already estimated
                
                set_probability_vector, p_vec_array, set_p_vec_array, leaf_likelihood_array = init_set_edges(set_probability_vector, set_p_vec_array, p_vec_array, set_p_vec_sample_matrix, leaf_likelihood_array, old_set_probabilities_matrix, phi_array, leaf_counts_array, layer_size, zero_contexts, suppress) 
                # sample a new tree status
                #### Sample a new value for every edge's indicator vector
                alpha_probs = (np.log(1-alpha), np.log(alpha))
                sample_new_indicator_gibbs(layer_size, alpha, alpha_probs, alpha_probabilities_array, indicator_array, phi_indicator_count_array, sigmas, phi_array, phi_probabilities_array, zero_contexts, suppress)
                #### Sample a new value of phi for every edge in this layer
                sample_new_phis_decoupled(layer, layer_size, total_leaves, phi_array, set_p_vec_array, p_vec_array, phi_probabilities_array, indicator_array, sigmas, phi_sigma_array, c, alpha, phi_batch_array, phi_sampling_alpha_array, phi_indicator_count_array, leaf_likelihood_array, leaf_counts_array, phi_accepted_array, post_burnin, adaptive_winddown, batch, zero_contexts, suppress)
                #### Sample a new value of alpha for this layer
                alpha = sample_new_alphas_beta(layer_size, alpha, alpha_probabilities_array, indicator_array, alpha_accepted_array, zero_contexts, suppress)
                #### Sample a new value of sigma for this layer
                if not set_sigma:
                    sigmas, phi_probabilities_array, sigma_sigma = sample_new_sigmas(layer_size, sigma_sigma, sigma_batch_array, c, sigmas, indicator_array, phi_array, phi_probabilities_array, sigma_accepted_array, post_burnin, adaptive_winddown, batch, zero_contexts, suppress)
               
            elif layer == 0:
                
                #### Sample a new value of phi for every edge in this layer
                sample_new_phis_decoupled_phi_naught(layer, layer_size, total_leaves, phi_array, p_vec_array, phi_sigma_array, phi_batch_array, leaf_likelihood_array, leaf_counts_array, phi_accepted_array, post_burnin, adaptive_winddown, batch, zero_contexts, suppress)
            # reset the batch
            if batch:
                phi_sigma_sample_matrix[batch-1] = np.copy(phi_sigma_array)
                phi_alpha_sample_matrix[batch-1] = np.copy(phi_sampling_alpha_array)
                phi_acceptance_batch_array[batch-1] = np.copy(phi_batch_array)
                batch = False
                phi_batch_array = np.zeros(layer_size * 3, dtype = np.float32)
                sigma_batch_array = np.array([0.0])
            
            ## set the new values for each matrix if it corresponds to one of the thinned samples
            if post_burnin and iteration % thinning_interval == 0:
                phi_sample_matrix[thinned_iteration] = np.copy(phi_array)
                rate_matrix[thinned_iteration] = np.copy(p_vec_array)
                indicator_sample_matrix[thinned_iteration] = np.copy(indicator_array)
                alpha_sample_matrix[thinned_iteration] = alpha
                if not set_sigma:
                    sigma_sample_matrix[thinned_iteration] = sigmas[1]
                set_probabilities_matrix[thinned_iteration] = set_probability_vector + phi_probabilities_array + alpha_probabilities_array
                thinned_iteration += 1
            
            # calculate the probabilities/likelihoods for each parameter
            total_phi_probability = np.sum(phi_probabilities_array)
            total_alpha_probability = np.sum(alpha_probabilities_array)
            total_leaf_likelihood = np.sum(leaf_likelihood_array)
            total_set_probability = np.sum(set_probability_vector)
            total_posterior = total_phi_probability + total_alpha_probability + total_set_probability + total_leaf_likelihood
            
            likelihood_matrix[iteration] = [total_phi_probability, total_alpha_probability, total_leaf_likelihood, total_set_probability, total_posterior]
            
            if not suppress:
                print("Total posterior from phi: ", total_phi_probability)
                print("Total posterior from alpha: ", total_alpha_probability)
                print("Total posterior from likelihood: ", total_leaf_likelihood)
                print("Total posterior from set probability: ", total_set_probability)
                print("JOINT POSTERIOR: ", total_posterior)
                print("Total iteration time: ", time.time() - sub_it_time_start)
        
        ## Once the layer has completed, dump the thinned trace to file

        likelihood_file = "{}/{}_{}_{}_rs{}_posterior_matrix.full_trace.layer_{}.npy".format(output_dir, pop, feature, dataset, random_seed, layer)
        alpha_file = "{}/{}_{}_{}_rs{}_alphas.burned_in.thinned.layer_{}.npy".format(output_dir, pop, feature, dataset, random_seed, layer)
        phis_file = "{}/{}_{}_{}_rs{}_phis.burned_in.thinned.layer_{}.npy".format(output_dir, pop, feature, dataset, random_seed, layer)
        indicator_file = "{}/{}_{}_{}_rs{}_indicator.burned_in.thinned.layer_{}.npy".format(output_dir, pop, feature, dataset, random_seed, layer)
        rate_matrix_file = "{}/{}_{}_{}_rs{}_rate_matrix.burned_in.thinned.layer_{}.npy".format(output_dir, pop, feature, dataset, random_seed, layer)
       
        np.save(likelihood_file, likelihood_matrix)
        np.save(alpha_file, alpha_sample_matrix)
        np.save(phis_file, phi_sample_matrix)
        np.save(indicator_file, indicator_sample_matrix)
        np.save(rate_matrix_file, rate_matrix)
        if not set_sigma:
            sigma_file = "{}/{}_{}_{}_rs{}_sigma_matrix.burned_in.thinned.layer_{}.npy".format(output_dir, pop, feature, dataset, random_seed, layer)
            np.save(sigma_file, sigma_sample_matrix)
                 
        old_set_probabilities_matrix = set_probabilities_matrix
        post_thin_samples = old_set_probabilities_matrix.shape[0]
        
        set_p_vec_sample_matrix = rate_matrix
        set_p_vec_sample_array = set_p_vec_sample_matrix[0]
        
        # Collect the index dict and dump to file
        index_dict = {}
        edge_index = 0
        for context in context_list:
            index_dict[edge_index] = str(context)
            edge_index += 1

        with open("{}/index_dict.layer_{}.json".format(output_dir, layer), 'w') as jFile:
            json.dump(index_dict, jFile)
        
        accepted_tuple = (sigma_accepted_array[0] / num_iterations, alpha_accepted_array[0] / num_iterations, phi_accepted_array[0] / (layer_size * 3 * num_iterations))
        write_layer_report(output_dir, layer, dataset, pop, feature, random_seed, num_iterations, burnin, accepted_tuple, starting_hyperparameters, thinning_interval, post_thin_samples, set_sigma, zero_init)
        

@njit()
def init_set_edges(set_probability_vector, current_set_p_vec_array, current_p_vec_array, set_p_vec_sample_matrix, leaf_likelihood_array, old_set_probabilities_matrix, phi_array, leaf_counts_array, layer_size, zero_contexts, suppress): 
    new_set_p_vec_array = current_set_p_vec_array
    total_parent_samples, total_parent_contexts, sub_indices = set_p_vec_sample_matrix.shape
    
    for parent_i in range(total_parent_contexts):
        
        random_sample_index = np.random.randint(0, total_parent_samples)
        new_parent_p_vec = set_p_vec_sample_matrix[random_sample_index, parent_i, :]
        new_parent_p_vec_array = np.zeros((4, 3))
        new_children_p_vecs = np.zeros((4, 3))
        for i in range(4):
            new_parent_p_vec_array[i] = new_parent_p_vec
            new_children_p_vecs[i] = new_parent_p_vec * np.exp(phi_array[parent_i * 4 + i])
        
        if np.max(new_children_p_vecs) >= 1 or np.max(np.sum(new_children_p_vecs, axis = 1)) >= 1:
            continue
        
        current_set_p_vec_array[parent_i * 4: parent_i * 4 + 4] = new_parent_p_vec_array
        current_p_vec_array[parent_i * 4: parent_i * 4 + 4] = new_children_p_vecs
        for i in range(4):
            set_probability_vector[parent_i * 4 + i] = old_set_probabilities_matrix[random_sample_index, parent_i, :] * 0.25
    
    leaf_likelihood_array = get_leaf_likelihood_array(layer_size, leaf_counts_array, current_p_vec_array, zero_contexts)
    return set_probability_vector, current_p_vec_array, current_set_p_vec_array, leaf_likelihood_array

############################################################
############################## ALPHAS ######################
############################################################

@njit()
def sample_new_alphas_beta(layer_size, alpha, alpha_probabilities_array, indicator_array, alpha_accepted_array, zero_contexts, suppress):

    ## ALPHAS are calculated using a Gibb's sampling step. Each alpha is drawn from a beta distribution

    indicator_sum = np.sum(indicator_array)
    # note there is a dummy index (-1) in zero contexts. Thus this context shouldn't be counted for indicator length
    indicator_length = layer_size * 3 - (len(zero_contexts) - 1)
    
    ## Get the current alpha posterior
    current_alpha_posterior = np.sum(alpha_probabilities_array)
    proposal_alpha = -1
    
    ### BETA SAMPLING
    # draw a new alpha from a beta distribution
    beta_alpha_param = indicator_sum + 1
    beta_beta_param = indicator_length - indicator_sum + 1
    proposal_alpha = float(np.random.beta(beta_alpha_param, beta_beta_param))
    # given the drawn value of alpha, calculate the posterior probability of the proposed alpha
    proposal_alpha_posterior = np.log(proposal_alpha)*indicator_sum + np.log(1-proposal_alpha)*(indicator_length-indicator_sum)
    # Use metropolis step to accept or reject (unnecessary given that this is a gibb's sampling step)
    accept = perform_metropolis_step_numba(proposal_alpha_posterior, current_alpha_posterior, suppress = True)
    # Update current values dependent on outcome of metropolis step
    
    if accept:
        update_alpha_probabilities(proposal_alpha, alpha_probabilities_array, indicator_array, layer_size, zero_contexts)
        alpha = proposal_alpha
        alpha_accepted_array[0] += 1
    return alpha

@njit(parallel = True)
def update_alpha_probabilities(alpha, alpha_probabilities_array, indicator_array, layer_size, zero_contexts):
    
    indicator_1_probability = np.log(alpha)
    indicator_0_probability = np.log(1-alpha)
    
    # update the probability of alpha according to indicator
    for i in prange(layer_size):
        for sub_index in range(3):
            if i in zero_contexts:
                alpha_probabilities_array[i][sub_index] = 0
            else:
                if indicator_array[i][sub_index] == 1:
                    alpha_probabilities_array[i][sub_index] = indicator_1_probability
                elif indicator_array[i][sub_index] == 0:
                    alpha_probabilities_array[i][sub_index] = indicator_0_probability


############################################################
############################## SIGMAS ######################
############################################################

@njit()
def sample_new_sigmas(layer_size, sigma_sigma, sigma_batch_array, c, sigmas, indicator_array, phi_array, phi_probabilities_array, sig_accepted_list, post_burnin, adaptive_winddown, batch, suppress):
    
    ## SIGMAS are sampled using a metropolis hastings step
    max_spike_sigma = 1
    # get the current value of sigma 
    spike_sigma, slab_sigma = sigmas
    # Propose a new value of sigma by drawing from a normal distribution
    proposal_slab_sigma = -1
    while proposal_slab_sigma <= 0 or (proposal_slab_sigma * c) >= max_spike_sigma:
        proposal_slab_sigma = np.random.normal(loc = slab_sigma, scale = sigma_sigma)
    proposal_spike_sigma = proposal_slab_sigma * c
    #### Evaluate the proposed sigma
    current_posterior = np.sum(phi_probabilities_array)
    prop_phi_probabilities_array  = get_vectorized_phi_probabilities(layer_size, proposal_slab_sigma, proposal_spike_sigma, phi_array, indicator_array)
    proposal_posterior = np.sum(prop_phi_probabilities_array)
    accept = perform_metropolis_step_numba(proposal_posterior, current_posterior, suppress = True)

    if accept:
        spike_sigma = proposal_spike_sigma
        slab_sigma = proposal_slab_sigma
        phi_probabilities_array = prop_phi_probabilities_array
        sig_accepted_list[0] += 1
        sigma_batch_array[0] += 1/BATCH
    #if batch and not post_burnin:
    if batch and not adaptive_winddown:
        lsi = np.log(sigma_sigma) / 2
        adjustment = 0.01
        acceptance_rate = sigma_batch_array[0]
        if acceptance_rate > 0.44:
            lsi = lsi + adjustment
        else:
            lsi = lsi - adjustment
        
        new_sigma_sigma = np.exp(2 * lsi)
        if new_sigma_sigma <= 0:
            new_sigma_sigma = sigma_sigma

        sigma_sigma = new_sigma_sigma
        

    return (spike_sigma, slab_sigma), phi_probabilities_array, sigma_sigma


@njit(parallel = True)
def get_vectorized_phi_probabilities(layer_size, proposal_slab_sigma, proposal_spike_sigma, phi_array, indicator_array):
    
    neg_log_slab_sigma = -np.log(proposal_slab_sigma)
    neg_log_spike_sigma = -np.log(proposal_spike_sigma)

    prop_phi_probabilities_array = np.zeros((layer_size, 3))
    
    for i in prange(layer_size):
        indicator = indicator_array[i]
        for j in range(3):
            sub_indicator = indicator[j]
            sub_phi = phi_array[i][j]
            if sub_indicator == 1:
                prop_phi_probabilities_array[i][j] = neg_log_slab_sigma - ((sub_phi/proposal_slab_sigma)**2)/2.0
            elif sub_indicator == 0:
                prop_phi_probabilities_array[i][j] = neg_log_spike_sigma - ((sub_phi/proposal_spike_sigma)**2)/2.0
 
    return prop_phi_probabilities_array


############################################################
############################## THETAS ######################
############################################################

@njit(parallel = True)
def sample_new_phis_decoupled(layer, layer_size, total_leaves, phi_array, set_p_vec_array, p_vec_array, phi_probabilities_array, indicator_array, sigmas, phi_sigma_array, c, alpha, phi_batch_array,  phi_sampling_alpha_array, phi_indicator_count_array, leaf_likelihood_array, leaf_counts_array, phi_accepted_array, post_burnin, adaptive_winddown, batch, zero_contexts, suppress):    
    accepted_array = np.zeros(3 * layer_size)
    
    for i in prange(layer_size):
        
        if i in zero_contexts:
            continue
        sub_indices = np.array([0,1,2])
        phi = phi_array[i]
        phi_probabilities = phi_probabilities_array[i]
        parent_p_vec = set_p_vec_array[i]
        p_vec = p_vec_array[i]
        leaf_likelihood = leaf_likelihood_array[i] 
        leaf_counts = leaf_counts_array[i]
        current_leaf_posterior = leaf_likelihood 
        
        ## make sure I don't sample in a particular order that might bias results
        np.random.shuffle(sub_indices)
        
        # for each phi sub index, propose and test a new value
        for sub_index in sub_indices:
            success = True
            current_posterior = current_leaf_posterior + phi_probabilities[sub_index]
            sigma = sigmas[indicator_array[i][sub_index]]
            prop_p_vec = np.copy(p_vec)
            sub_parent_p_vec = parent_p_vec[sub_index]
            phi_sigma = phi_sigma_array[i*3 + sub_index]
            phi_alpha = phi_sampling_alpha_array[i*3 + sub_index]
            phi_alpha = 0.5
            # first sample from the phis
            ## first flip a coin to decide which distribution to sample from
            dist = indicator_array[i][sub_index]
            
            prop_sub_phi = np.random.normal(loc = phi[sub_index], scale = phi_sigma * c**(1-dist))
            prop_sub_p_vec = np.exp(prop_sub_phi) * sub_parent_p_vec
            
            prop_p_vec[sub_index] = prop_sub_p_vec
            success = False
            if prop_sub_p_vec > 0 and np.sum(prop_p_vec) < 1:
                success = True
            
            fail_count = 0
            while not success:
                
                prop_sub_phi = np.random.normal(loc = phi[sub_index], scale = phi_sigma * c**(1-dist))
                prop_sub_p_vec = np.exp(prop_sub_phi) * sub_parent_p_vec
                prop_p_vec[sub_index] = prop_sub_p_vec
                
                if prop_sub_p_vec > 0 and np.sum(prop_p_vec) < 1:
                    success = True
                fail_count += 1
                # try to find a phi value that works. If it fails 5 times, move on so we don't get stuck
                if fail_count > 5:
                    break
            if not success:
                continue
            '''
            if np.sum(prop_p_vec) < 1:
                lower_bound = 0
                p_available = 1 - (np.sum(p_vec) - p_vec[sub_index])
                upper_bound = np.log(p_available/sub_parent_p_vec)
                #prop_sub_phi =  
            '''    
            # get the new phi probability
            prop_sub_phi_probability = -np.log(sigma) - ((prop_sub_phi/sigma)**2)/2.0
            
            ## I can then calculate likelihoods and accept or reject phis
            ## first find the likelihood
            prop_leaf_posterior = get_edge_likelihood_numba(prop_p_vec, leaf_counts)        
            proposal_posterior = prop_sub_phi_probability + prop_leaf_posterior
            ## Next perform the metropolis step
            mh_suppress = True
            accept = perform_metropolis_step_numba(proposal_posterior, current_posterior, mh_suppress)
            
            # update the values for this sampled phi if accepted
            if accept:
                current_leaf_posterior = prop_leaf_posterior
                leaf_likelihood_array[i] = current_leaf_posterior
                phi[sub_index] = prop_sub_phi
                p_vec[sub_index] = prop_sub_p_vec
                phi_probabilities[sub_index] = prop_sub_phi_probability
                accepted_array[i + sub_index] = 1
               
                phi_batch_array[i*3 + sub_index] += np.float32(1/BATCH)
                    
            #if batch and not post_burnin:
            if batch and not adaptive_winddown:
                lsi = np.log(phi_sigma) / 2
                adjustment = 0.01
                
                acceptance_rate = phi_batch_array[i*3 + sub_index]
                if acceptance_rate > 0.44:
                    lsi = lsi + adjustment
                else:
                    lsi = lsi - adjustment
        
                new_phi_sigma = np.exp(2 * lsi)
                if new_phi_sigma <= 0:
                    new_phi_sigma = phi_sigma
                elif new_phi_sigma >= 1.5:
                    new_phi_sigma = phi_sigma

                phi_sigma = new_phi_sigma
            
                phi_sigma_array[i*3 + sub_index] = new_phi_sigma
                
                indicator_count = phi_indicator_count_array[i*3 + sub_index]
                new_phi_alpha = np.float32((500 + indicator_count) / (1000  + batch * BATCH))
                
                phi_sampling_alpha_array[i*3 + sub_index] = new_phi_alpha
    
    phi_accepted_array[0] += np.sum(accepted_array)

@njit()
def sample_new_phis_decoupled_phi_naught(layer, layer_size, total_leaves, phi_array, p_vec_array, phi_sigma_array, phi_batch_array, leaf_likelihood_array, leaf_counts_array, phi_accepted_array, post_burnin, adaptive_winddown, batch, zero_contexts, suppress):
     
    # for the case where you are at the root of the tree. The sampling procedure is slightly different
    p_vec = p_vec_array[0]
    leaf_likelihood = leaf_likelihood_array[0] 
    leaf_counts = leaf_counts_array[0]
    current_leaf_posterior = leaf_likelihood 

    sub_indices = np.array([0,1,2])
    ## make sure I don't sample in a particular order that might bias results
    np.random.shuffle(sub_indices)
    any_accept = False
    accepted_array = np.zeros(3)
    # for each phi sub index, propose and test a new value
    count = -1
    for i in range(layer_size):
        if i in zero_contexts:
            continue
        phi = phi_array[i]
        p_vec = p_vec_array[i]
        leaf_likelihood = leaf_likelihood_array[i]
        leaf_counts = leaf_counts_array[i]
        current_leaf_posterior = leaf_likelihood
        
        for sub_index in sub_indices:
            count += 1
            current_posterior = current_leaf_posterior
            
            phi_sigma = phi_sigma_array[i*3 + sub_index]
            prop_p_vec = np.copy(p_vec)

            # first sample from the phis
            prop_sub_phi = 0
            prop_sub_p_vec = 0
            success = False
            while not success:
                prop_sub_phi = np.random.normal(loc = phi[sub_index], scale = phi_sigma)
                prop_sub_p_vec = prop_sub_phi
                prop_p_vec[sub_index] = prop_sub_p_vec
                if np.min(prop_p_vec) > 0 and np.sum(prop_p_vec) < 1.0:
                    success = True

            ## I can then calculate likelihoods and accept or reject phis
            ## first find the likelihood
            proposal_posterior = get_edge_likelihood_numba(prop_p_vec, leaf_counts)        
            ## Next perform the metropolis step
            mh_suppress = True
            accept = perform_metropolis_step_numba(proposal_posterior, current_posterior, mh_suppress)
            # update the values for this sampled phi if accepted
            if accept:
                any_accept = True
                current_leaf_posterior = proposal_posterior
                phi[sub_index] = prop_sub_phi
                p_vec[sub_index] = prop_sub_phi
                leaf_likelihood_array[i] = current_leaf_posterior 
                accepted_array[count] = 1
                # adjust the value of the proposal sigma
                phi_batch_array[i*3 + sub_index] += np.float32(1/BATCH)
                     
            #if batch and not post_burnin:
            if batch and not adaptive_winddown:
                lsi = np.log(phi_sigma) / 2
                adjustment = 0.01
                 
                acceptance_rate = phi_batch_array[i*3 + sub_index]
                if acceptance_rate > 0.44:
                    lsi = lsi + adjustment
                else:
                    lsi = lsi - adjustment
        
                new_phi_sigma = np.exp(2 * lsi)
                if new_phi_sigma <= 0:
                    new_phi_sigma = phi_sigma
                elif new_phi_sigma >= 1.5:
                    new_phi_sigma = phi_sigma
                phi_sigma_array[i*3 + sub_index] = new_phi_sigma

        phi_accepted_array[0] += np.sum(accepted_array)


################################################################
############################## INDICATORS ######################
################################################################


@njit(parallel = True)
def sample_new_indicator_gibbs(layer_size, alpha, alpha_probs, alpha_probabilities_array, indicator_array, phi_indicator_count_array, sigmas, phi_array, phi_probabilities_array, zero_contexts, suppress):
    
    spike_alpha_prob, slab_alpha_prob = alpha_probs
    spike_sigma, slab_sigma = sigmas
    
    for i in prange(layer_size):
        if i in zero_contexts:
            continue

        indicator = indicator_array[i]
        
        phi = phi_array[i]
        current_posterior = 0
        for sub_index in range(3):
            sub_phi = phi[sub_index]
            sub_indicator = indicator[sub_index]
            slab_phi_probability = 0
            spike_phi_probability = 0
            if sub_indicator == 1:
                slab_phi_probability = phi_probabilities_array[i][sub_index]
                spike_phi_probability = -np.log(spike_sigma) - ((sub_phi/spike_sigma)**2)/2.0
            else:
                spike_phi_probability = phi_probabilities_array[i][sub_index]
                slab_phi_probability = -np.log(slab_sigma) - ((sub_phi/slab_sigma)**2)/2.0
            
            spike_posterior = np.exp(spike_alpha_prob + spike_phi_probability)
            slab_posterior = np.exp(slab_alpha_prob + slab_phi_probability)

            sampling_p = slab_posterior / (slab_posterior + spike_posterior)
            new_sub_indicator = int(np.random.binomial(1, sampling_p, 1)[0])
            
            indicator_array[i][sub_index] = new_sub_indicator
            if new_sub_indicator == 1:
                phi_probabilities_array[i][sub_index] = slab_phi_probability
                alpha_probabilities_array[i][sub_index] = slab_alpha_prob
            else:
                phi_probabilities_array[i][sub_index] = spike_phi_probability
                alpha_probabilities_array[i][sub_index] = spike_alpha_prob
            
            phi_indicator_count_array[3*i + sub_index] += indicator_array[i][sub_index]

#############################################################################
############################## COMPLEMENTARY FUNCTIONS ######################
#############################################################################

@njit()
def perform_metropolis_step_numba(proposal_posterior, current_posterior, suppress = True):
    
    if not suppress:
        print("Proposal posterior: ",proposal_posterior)
        print("Current posterior:  ",current_posterior)

    if proposal_posterior >= current_posterior:
        if not suppress:
            print("ratio is > 1")
            print("ACCEPT")
            print('+++++++++++')
        return True

    ratio = np.exp(proposal_posterior - current_posterior)
    random_sample = np.random.uniform(0, 1)
    if not suppress:
        print("MH ratio: ", ratio)
        print("Sample:   ", random_sample)
        if random_sample < ratio:
            print("ACCEPT")
        else:
            print("REJECT")
        print('+++++++++++')
    if random_sample < ratio:
        return True
    else:
        return False

@njit()
def get_edge_likelihood_numba(p_vec, leaf_counts):

    likelihood = np.sum(leaf_counts[0::4]) * np.log(1.0-np.sum(p_vec)) + \
                 np.sum(leaf_counts[1::4]) * np.log(p_vec[0]) + \
                 np.sum(leaf_counts[2::4]) * np.log(p_vec[1]) + \
                 np.sum(leaf_counts[3::4]) * np.log(p_vec[2])
    return likelihood

 
@njit(parallel = True) 
def get_leaf_likelihood_array(layer_size, leaf_counts_array, p_vec_array, zero_contexts):

    leaf_likelihood_array = np.ones(layer_size)
    for i in prange(layer_size):
        if i in zero_contexts:
            continue
        p_vec = p_vec_array[i]
        leaf_counts = leaf_counts_array[i]

        leaf_likelihood = get_edge_likelihood_numba(p_vec, leaf_counts)    
        
        leaf_likelihood_array[i] = leaf_likelihood

    return leaf_likelihood_array

@njit()
def get_set_p_vec_samples(layer_size, rate_matrix, post_thin_samples):
    
    next_layer_size = layer_size * 4
    set_p_vec_sample_array = np.zeros((post_thin_samples, next_layer_size, 3))
    for parent_index in range(layer_size):
        parent_context_set_p_vec_samples = rate_matrix[:, parent_index]
        for index_mod in range(4):
            array_index = index_mod + (parent_index * 4)
            set_p_vec_sample_array[:, array_index] = parent_context_set_p_vec_samples
    
    return set_p_vec_sample_array

@njit()
def project_set_p_vec_sample_array(parent_layer_set_p_vec_sample_array, layer_size):
    
    set_p_vec_sample_array = np.zeros((layer_size, 3))
    parent_index = 0
    for parent_set_p_vec_sample in parent_layer_set_p_vec_sample_array:
        for index_mod in range(4):
             array_index = index_mod + (parent_index * 4)
             set_p_vec_sample_array[array_index] = parent_set_p_vec_sample

        parent_index += 1

    return set_p_vec_sample_array

def prep_leaf_count_dict(config_dict, pop, feature, max_mer, dataset):
        
    pop_feature_config_dict = config_dict[pop][feature]

    mer_string = str(max_mer)+"mer"
        
    count_file = pop_feature_config_dict[mer_string][dataset]

    with open(count_file) as jFile:
        leaf_count_dict = json.load(jFile)
        
    return leaf_count_dict


def initialize_layer_tree_edge_list(layer, layer_size, max_mer, old_context_list, sigmas, alpha_probs, set_p_vec_sample_array, leaf_count_dict, first_sample, random_seed, zero_init, oppo_asymmetry = False):
    
    if random_seed:
        np.random.seed(int(random_seed))
        random.seed(int(random_seed))
    ## Init all the returned arrays
    phi_array = np.zeros((layer_size, 3), dtype=np.float32)
    phi_probabilities_array = np.zeros((layer_size, 3))
    p_vec_array = np.ones((layer_size, 3), dtype=np.float32)
    indicator_array = np.zeros((layer_size, 3), dtype=np.int32)
    alpha_probabilities_array = np.zeros((layer_size, 3))
    context_list = [None for x in range(layer_size)]
    leaf_count_components = 4**(int(max_mer) - int(layer))
    leaf_count_array = np.ones((layer_size, leaf_count_components))
    
    # holds indices for contexts that have not been seen in the dataset
    zero_contexts = [-1]

    spike_sigma, slab_sigma = sigmas
    spike_alpha_prob, slab_alpha_prob = alpha_probs 

    # check if this is the first layer
    if layer == 0:
        context_list = np.array(['A', 'C'])
        array_index = -1
        for context in context_list:
            array_index += 1
            init_phi = np.random.uniform(low = 0.0, high = 0.02, size = 3)
            init_p_vec = init_phi
                
            phi_array[array_index] = init_phi
            p_vec_array[array_index] = init_p_vec
            
            # get leaf_counts
            leaf_contexts = get_leaf_contexts(context, max_mer, oppo_asymmetry)
            
            leaf_counts = None
            if context == 'C':
                leaf_counts = get_adjusted_leaf_counts(leaf_contexts, leaf_count_dict).astype(np.int32)
            else:
                leaf_counts = np.array([np.array(leaf_count_dict[x][0:4]) for x in leaf_contexts]).flatten().astype(np.int32)
            
            leaf_count_array[array_index] = leaf_counts
            if np.sum(leaf_counts) == 0:
                print("ERROR: {} has zero contexts in this dataset")
                help()
                
    else:
        context_size = layer
        odd_bool = context_size % 2
        parent_index = -1
        for parent_context in old_context_list:
            C_bool = True
            
            central_nuc_index = int(context_size/2)
            if not odd_bool and not oppo_asymmetry:
                central_nuc_index = central_nuc_index - 1
                
            central_nuc = parent_context[central_nuc_index]
            if central_nuc == 'C':
                C_bool = True
            elif central_nuc == 'A':
                C_bool = False
            else:
                print("Error: non A or C central nuc")
                sys.exit()
            parent_index += 1
            index_mod = 0
            zero_contexts_bool = False
            for nuc in ['A', 'C', 'G', 'T']:
                array_index = index_mod + (parent_index * 4)
                
                
                child_context = nuc + parent_context
                if (not oppo_asymmetry and odd_bool) or (oppo_asymmetry and not odd_bool):
                    child_context = parent_context + nuc
                    
                context_list[array_index] = child_context
                
                # get leaf counts and check for zero contexts
                leaf_contexts = get_leaf_contexts(child_context, max_mer, oppo_asymmetry)
                leaf_counts = []
                if C_bool:
                    leaf_counts = get_adjusted_leaf_counts(leaf_contexts, leaf_count_dict).astype(np.int32)
                else:
                    leaf_counts = np.array([np.array(leaf_count_dict[x][0:4]) for x in leaf_contexts]).flatten().astype(np.int32)
                
                leaf_count_array[array_index] = leaf_counts
                
                if np.sum(leaf_counts) == 0:
                    zero_contexts.append(array_index)
                    zero_contexts_bool = True
                    
                # Init phi
                init_phi = np.array([0, 0, 0])
                init_p_vec = 0
                init_indicator = np.array([0, 0, 0])
                
    
                parent_p_vec = set_p_vec_sample_array[array_index, :]
                success = False
                while not success:
                    if not zero_init and not zero_contexts_bool:
                        init_phi = np.random.uniform(low = -0.7, high = 0.7, size = 3)
                    init_p_vec = parent_p_vec * np.exp(init_phi)
                    # make sure this p_vec is valid
                    if np.sum(init_p_vec) < 1 and min(init_p_vec) > 0:
                        success = True 
                
                phi_array[array_index] = init_phi
                p_vec_array[array_index] = init_p_vec
                
                # Init indicator
                if not zero_init and not zero_contexts_bool:
                    init_indicator = np.random.randint(2, size = 3)
                    indicator_array[array_index] = init_indicator

                # Init sigma array, phi probabilities, alpha probabilities
                count = 0
                for j in init_indicator:
                    sub_phi_probability = 0
                    alpha_prob = 0
                    if not zero_contexts_bool:
                        sigma = slab_sigma
                        alpha_prob = slab_alpha_prob
                        if j == 0:
                            sigma = spike_sigma
                            alpha_prob = spike_alpha_prob
                        sub_phi_probability = -np.log(sigma) - ((init_phi[count]/sigma)**2)/2.0
                    
                    phi_probabilities_array[array_index][count] = sub_phi_probability
                    alpha_probabilities_array[array_index][count] = alpha_prob
                    count += 1

                        
                index_mod += 1
    
    return phi_array, p_vec_array, indicator_array, context_list, leaf_count_array, phi_probabilities_array, alpha_probabilities_array, np.array(zero_contexts)


def get_adjusted_leaf_counts(leaf_contexts, leaf_count_dict):

    leaf_counts = np.zeros(len(leaf_contexts) * 4)
    context_count = 0
    for context in leaf_contexts:
        counts = leaf_count_dict[context]
        new_counts = np.array([counts[1], counts[0], counts[2], counts[3]])
        leaf_counts[context_count*4:context_count*4 + 4] = new_counts
        context_count += 1
    
    return leaf_counts

def get_leaf_contexts(context, max_mer, oppo_asymmetry):
    leaf_contexts = []
    context_length = len(context)
    
    odd_max_bool = max_mer % 2

    nucleotides_to_add = int(max_mer - context_length)
    odd_bool = 0
    if odd_max_bool and nucleotides_to_add % 2 and not oppo_asymmetry:
        odd_bool = 1
    elif not odd_max_bool and nucleotides_to_add % 2 and oppo_asymmetry:
        odd_bool = 1
    left_flank_length = int(nucleotides_to_add / 2) + odd_bool
    
    for combination in product('ACGT', repeat=nucleotides_to_add):
        bases = ''.join(combination)
        if left_flank_length == 0:
            leaf_context = context + bases
        else:
            leaf_context = bases[0:left_flank_length] + context + bases[left_flank_length:]
        leaf_contexts.append(leaf_context)
        
    return leaf_contexts


def get_final_nodes(repped_nodes, total_nodes):
    

    final_leaf_nodes = []
    layer_nodes = repped_nodes[:]
    
    stop_bool = False
    while not stop_bool:
        
        new_layer_nodes = []
        for node in layer_nodes:
            node_children = [4 * int(node) + mod for mod in range(1,5)]
            if node_children[0] >= total_nodes:
                # max level reached
                final_leaf_nodes.append(node)
            else:
                new_layer_nodes = new_layer_nodes + node_children
        if len(new_layer_nodes) == 0:
            stop_bool = True
        else:
            layer_nodes = new_layer_nodes[:]

    return final_leaf_nodes


def write_layer_report(output_dir, layer, dataset, pop, feature, random_seed, num_iterations, burnin, accepted_tuple, starting_hyperparameters, thinning_interval, post_thin_samples, set_sigma, zero_init):
    
    output_file = output_dir + "/{}_layer_{}_report.{}.{}.rs{}.txt".format(dataset, layer, pop, feature, random_seed)

    with open(output_file, 'w') as out:
        out.write("Population: {}\nFeature: {}\nLayer: {}\nData partition: {}\nRandom seed: {}\nLayer total iterations: {}\nLayer burn in: {}\n".format(pop, feature, layer, dataset, random_seed, num_iterations, burnin))
        out.write("Thin number: {}\nPost-thinning samples: {}\nzero init: {}\n".format(thinning_interval, post_thin_samples, zero_init))
        out.write("--------------\n")
        out.write("Jump information\nProposal Sigma: {}\nSigma sigma: {}\nalpha sigma: {}\nIndicator sampling method: {}\n".format(starting_hyperparameters[0], starting_hyperparameters[1], starting_hyperparameters[5], starting_hyperparameters[6]))
        out.write("Fraction sub phis accepted: {}\nFraction sigmas accepted: {}\nFraction alphas accepted: {}\n".format(accepted_tuple[2], accepted_tuple[0], accepted_tuple[1]))
        out.write("--------------\n")
        out.write("Set starting conditions\nc: {}\nset_sigma: {}\nslab sigma: {}\nalpha: {}\n".format(starting_hyperparameters[2], set_sigma, starting_hyperparameters[3],starting_hyperparameters[4]))
        #out.write("Timings\nAverage time per iteration: {}\nAvg init set edges time: {}\nAvg indicator sampling time: {}\nAvg phi sampling time: {}\nAvg alpha sampling time: {}\nAvg sigma sampling time: {}\n".format(timing_tuple[0], timing_tuple[1], timing_tuple[2], timing_tuple[3], timing_tuple[4], timing_tuple[5]))
        out.write("--------------")


#######################################################################################################################################################
#cProfile.run('main(sys.argv[1:])')

if __name__ == "__main__":
    main(sys.argv[1:])
