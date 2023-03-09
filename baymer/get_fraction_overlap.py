#!/usr/bin/Python

# Created by: Christopher J Adams 9/13/2022
# 

###############################################################################
###
### This script will calculate the percent difference between two datasets parameter distributions
###
###############################################################################


## NOTE Assumes symmetric-ish counts

#import cProfile
import sys
import getopt
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import yaml
import json
import multiprocessing as mp
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
from numba import njit


def help(exit_num=1):
    print("""-----------------------------------------------------------------
ARGUMENTS
    -a => <yaml> config file dataset 1 REQUIRED
    -b => <yaml> config file dataset 2 REQUIRED
    -o => <dir> output directory REQUIRED
    -s => <int> start layer OPTIONAL
    -r => <float> adjustment to make in first dataset OPTIONAL Default: 1.0
    -x => <bool> presence indicates to use the opposite even symmetry OPTIONAL
""")
    sys.exit(exit_num)

###############################################################################
###########################  COLLECT AND CHECK ARGUMENTS  #####################
###############################################################################

## GLOBAL VARS

## MAIN ##

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "a:b:o:s:r:x")
    except getopt.GetoptError:
        print("Error: Incorrect usage of getopts flags!")
        help()

    options_dict = dict(opts)

    ## Required arguments
    try:
        a_config_file = options_dict['-a']
        b_config_file = options_dict['-b']
        output_dir = options_dict['-o']

    except KeyError:
        print("Error: One of your required arguments does not exist.")
        help()
    start_layer = int(options_dict.get('-s', 0))
    adj_ratio = float(options_dict.get("-r", 1.0))
    opposite_symmetry = options_dict.get("-x", False)
    if opposite_symmetry == "":
        opposite_symmetry = True
    print("Acceptable Inputs Given")
    
    

    driver(a_config_file, b_config_file, output_dir, start_layer, adj_ratio, opposite_symmetry)


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(a_config_file, b_config_file, output_dir, start_layer, adj_ratio, opposite_symmetry):
    
    a_config_dict = yaml.load(open(a_config_file, 'r'), Loader=yaml.SafeLoader)
    b_config_dict = yaml.load(open(b_config_file, 'r'), Loader=yaml.SafeLoader)
    
    max_layer = min([int(a_config_dict['max_layer']), int(b_config_dict['max_layer'])])
    
    a_pop = a_config_dict['pop']
    b_pop = b_config_dict['pop']
    a_feature = a_config_dict['feature']
    b_feature = b_config_dict['feature']
    a_posterior_dir = a_config_dict['posterior_dir']
    b_posterior_dir = b_config_dict['posterior_dir']
    a_random_seeds = a_config_dict['random_seeds']
    b_random_seeds = b_config_dict['random_seeds']
    a_dataset = a_config_dict['dataset']
    b_dataset = b_config_dict['dataset']
    pops = (a_pop + "_" + a_dataset, b_pop + "_" + b_dataset)
    a_c = a_config_dict['c']
    b_c = b_config_dict['c']
    num_layers = max_layer - start_layer + 1
    
    master_df = None
    
    for layer in range(max_layer + 1):
        merged_layer_df = None
 
        mer_string = str(layer + 1) + "mer"
        a_posterior_dict = {}
        b_posterior_dict = {}
        
        # gather layer data
        index_dict = a_posterior_dir + "index_dict.layer_" + str(layer) + ".json"

        index_context_dict = json.load(open(index_dict, 'r'))

        # first thetas and p_vec
        a_theta_matrix, a_p_matrix = get_data_matrices(a_posterior_dir, a_pop, a_feature, a_dataset, a_random_seeds, layer) 
        b_theta_matrix, b_p_matrix = get_data_matrices(b_posterior_dir, b_pop, b_feature, b_dataset, b_random_seeds, layer) 
        
        if layer == 0:
           theta_adj_ratio = adj_ratio
        else:
            theta_adj_ratio = 1.0

        theta_df = parallel_compare_distributions_driver(layer, a_theta_matrix, b_theta_matrix, index_context_dict, pops, theta_adj_ratio, theta = True, opposite_symmetry = opposite_symmetry)
        p_vec_df = parallel_compare_distributions_driver(layer, a_p_matrix, b_p_matrix, index_context_dict, pops, adj_ratio, opposite_symmetry = opposite_symmetry)
        
        merged_layer_df = theta_df.merge(p_vec_df, on = ["layer", "context_mutation", "mut_type"])
        layer_output_csv = "{}/{}mer.{}_{}_{}.{}_{}_{}.joint_theta_p.fraction_overlap.csv".format(output_dir, layer + 1, a_pop, a_dataset, a_feature, b_pop, b_dataset, b_feature)
        merged_layer_df.to_csv(layer_output_csv, index = False)
        if layer == 0:
            master_df = merged_layer_df
        else:
            master_df = master_df.append(merged_layer_df)
    
    master_output_csv = "{}/all_mers.{}_{}_{}.{}_{}_{}.joint_theta_p.fraction_overlap.csv".format(output_dir, a_pop, a_dataset, a_feature, b_pop, b_dataset, b_feature)
    
    master_df.to_csv(master_output_csv, index = False)
    

    

def get_data_matrices(posterior_dir, pop, feature, dataset, random_seeds, layer): 
    theta_matrix = []
    p_matrix = []
    first_seed = True
    for random_seed in random_seeds:
        theta_chain_matrix_file = "{}{}_{}_{}_rs{}_thetas.burned_in.thinned.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
        theta_chain_matrix = np.load(theta_chain_matrix_file)
        p_vec_file = "{}{}_{}_{}_rs{}_rate_matrix.burned_in.thinned.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
        p_chain_matrix = np.load(p_vec_file)
        if first_seed:
            theta_matrix = theta_chain_matrix
            p_matrix = p_chain_matrix
            first_seed = False
        else:
            theta_matrix = np.append(theta_matrix, theta_chain_matrix, axis = 0)
            p_matrix = np.append(p_matrix, p_chain_matrix, axis = 0)
    
    return theta_matrix, p_matrix


def parallel_compare_distributions_driver(layer, a_matrix, b_matrix, index_context_dict, pops, adj_ratio = 1, theta = False, opposite_symmetry = False):
    
    total_contexts = len(a_matrix[0])
    
    pool = mp.Pool(40)
    frac_overlap_results = [pool.apply_async(parallelized_context_dist_overlaps, args = (index, a_matrix[:, index, :], b_matrix[:, index, :], adj_ratio)) for index in range(total_contexts)]
    
    dist_type = "p"
    if theta:
        dist_type = "theta"
    
    center_nuc_index = int((layer + 1)/2)
    if layer % 2 == 1 and not opposite_symmetry:
        center_nuc_index = center_nuc_index - 1

    mut_nuc_index_look_up = ['A', 'C', 'G', 'T']
    
    df_list_of_lists = []
    df_column_names = ["layer", "context_mutation", "mut_type", "fraction_overlap_"+dist_type, pops[0] + "_mean_" + dist_type, pops[1] + "_mean_" + dist_type]    

    for result in frac_overlap_results:
        
        context_index, overlap_1, a_mean_1, b_mean_1, overlap_2, a_mean_2, b_mean_2, overlap_3, a_mean_3, b_mean_3  = result.get()
        context_string = index_context_dict[str(context_index)]
        center_nuc = context_string[center_nuc_index]
        mut_bases = ["C", "G", "T"]
        
        if center_nuc == "C":
            mut_bases = ["A", "G", "T"]
            try:
                cpg_test_nuc = context_string[center_nuc_index + 1]
                if cpg_test_nuc == "G":
                    center_nuc = "CpG"
            except IndexError:
                pass
        
        df_list_of_lists.append([layer, context_string + ">" + mut_bases[0], center_nuc + ">" + mut_bases[0], overlap_1, a_mean_1, b_mean_1])
        df_list_of_lists.append([layer, context_string + ">" + mut_bases[1], center_nuc + ">" + mut_bases[1], overlap_2, a_mean_2, b_mean_2])
        df_list_of_lists.append([layer, context_string + ">" + mut_bases[2], center_nuc + ">" + mut_bases[2], overlap_3, a_mean_3, b_mean_3])
        
    df = pd.DataFrame(data = df_list_of_lists, columns = df_column_names)
    
    return df
    

def parallelized_context_dist_overlaps(index, a_row, b_row, adj_ratio):
    
    context_list = [index]
    
    if float(adj_ratio) != 1.0:
        a_row = a_row * adj_ratio
    
    for mut_index in [0, 1, 2]:
        a_sub_posterior = a_row[:, mut_index]
        b_sub_posterior = b_row[:, mut_index]
        
        a_mean = np.mean(a_sub_posterior)
        b_mean = np.mean(b_sub_posterior)

        lower_bound = min([np.min(a_sub_posterior), np.min(b_sub_posterior)])
        upper_bound = max([np.max(a_sub_posterior), np.max(b_sub_posterior)])
        
        
        a_dist = gaussian_kde(a_sub_posterior)
        
        b_dist = gaussian_kde(b_sub_posterior)
        scipy_fraction_overlap = scipy_gaussian_kernel_estimate_driver(a_dist, b_dist, lower_bound, upper_bound)

        context_list.extend([scipy_fraction_overlap, a_mean, b_mean])
    
    return context_list


def scipy_gaussian_kernel_estimate_driver(a_dist, b_dist, lower_bound, upper_bound):
    
    hist_sequences = np.linspace(lower_bound, upper_bound, 1000)
    hist_sequences = np.atleast_2d(hist_sequences).astype("float32")
    points = hist_sequences.T.astype("float32")
    
    a_dataset = a_dist.dataset.T.astype("float32")
    weights = a_dist.weights[:, None].astype("float32")
    inv_cov = a_dist.inv_cov.astype("float32")
    a_vals = numba_enabled_kernel_density_estimate(a_dataset, weights, points, inv_cov)

    b_dataset = b_dist.dataset.T.astype("float32")
    weights = b_dist.weights[:, None].astype("float32")
    inv_cov = b_dist.inv_cov.astype("float32")
    b_vals = numba_enabled_kernel_density_estimate(b_dataset, weights, points, inv_cov)
    comb_f_x = np.maximum(a_vals, b_vals)
    overlap_f_x = np.minimum(a_vals, b_vals)
    
    return(np.mean(overlap_f_x)/np.mean(comb_f_x))

@njit()
def numba_enabled_kernel_density_estimate(dataset, weights, points, inv_cov):
    
    n = dataset.shape[0]
    d = 1
    m = points.shape[0]
    p = weights.shape[1]

    # rescale the data
    whitening = np.linalg.cholesky(inv_cov)
    dataset_ = np.dot(dataset, whitening)
    points_ = np.dot(points, whitening)
    # evaluate the normalization
    norm = np.power((2 * np.pi), (-1 / 2)) * whitening[0, 0]
    # create the result array and evaluate the weighted sum
    estimate = np.zeros(m)
    residual_M = dataset_.reshape((-1, 1)) - points_.reshape((1, -1))
    residual_M_squared = residual_M**2
    arg_M_final = weights * np.exp(-residual_M_squared/2) * norm
    estimate = np.sum(arg_M_final, axis = 0)
    
    return estimate


#######################################################################################################################################################
#cProfile.run('main(sys.argv[1:])')
         
if __name__ == "__main__":
    main(sys.argv[1:])
