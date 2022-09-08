#!/usr/bin/Python

# Created by: Christopher J Adams 9/8/2022
# 

###############################################################################
###
### This script will convert a variant list into a count dict given certain ac specifications
###
###############################################################################


## NOTE Assumes symmetric-ish counts

import cProfile
import sys
import getopt
import os
import re
import yaml
import json
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from numba import njit, prange

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'modules'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'model_generation'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'collapse_model'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'compare_models'))

import projection_functions
import likelihood_ratio_test_class
import count_utils
import wrapper_full_mer_model as rate_dict_generator
import fold_rate_dicts
import exhaustive_mer_expansion
import collapsing_algorithm_classes
import general_utils
import simulation_functions
import bootstrap_functions
import rate_comparisons


def help(exit_num=1):
    print("""-----------------------------------------------------------------
ARGUMENTS
    -m => <csv> mutations csv REQUIRED
    -c => <json> contexts csv REQUIRED
    -o => <dir> output dir REQUIRED
    -p => <str> pop REQUIRED
    -f => <str> feature REQUIRED
    -d => <str> dataset REQUIRED
    --max-af => <float> maximum allele frequency which is considered OPTIONAL Default: 0.85
ASSUMPTIONS
    * Both datasets are from the same feature
    * Only includes autosomal data
    * the dataset specified must be either EVEN, ODD, or ALL
""")
    sys.exit(exit_num)

###############################################################################
###########################  COLLECT AND CHECK ARGUMENTS  #####################
###############################################################################

## GLOBAL VARIABLES

## MAIN ##

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "m:c:o:p:f:d:", ["max-af=", "quality="])
                                                              
    except getopt.GetoptError:
        print("Error: Incorrect usage of getopts flags!")
        help()

    options_dict = dict(opts)
    
    ## Required arguments
    try:
        output_dir = options_dict['-o']
        mutation_count_file = options_dict['-m']
        context_count_file = options_dict['-c']
        feature = options_dict['-f']
        pop = options_dict['-p']
        dataset = options_dict['-d']
    
    except KeyError:
        print("Error: One of your required arguments does not exist.")
        help()
    
    # optional arguments
    max_af = float(options_dict.get("--max-af", 0.85))
    quality = options_dict.get("--quality", False)
    
    print("Acceptable Inputs Given")

    driver(mutation_count_file, context_count_file, feature, pop, dataset, output_dir, max_af, quality)


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(mutation_count_file, context_count_file, feature, pop, dataset, output_dir, max_af, quality):
    


    # prepare the master count dict from the context count file
    context_count_dict = prepare_count_json_from_csv(context_count_file, dataset)

    # mutation count data
    mutation_count_df = pd.read_csv(mutation_count_file)
    if dataset == "EVEN":
        mutation_count_df = mutation_count_df.loc[mutation_count_df["even_odd_bool"] == 0]
    elif dataset == "ODD":
        mutation_count_df = mutation_count_df.loc[mutation_count_df["even_odd_bool"] == 1]
    
    mutation_count_df['AF'] = mutation_count_df['AC'] / mutation_count_df['AN']
    mutation_count_df = mutation_count_df.loc[mutation_count_df["AF"] <= max_af]
    if quality is not False:
        mutation_count_df = mutation_count_df.loc[mutation_count_df["quality_score"] >= float(quality)]
    total_muts = mutation_count_df.count()[0]
    print("variants counted from mutation file: ", total_muts)
    
    min_mer = 1
    max_mer = 9

    count_dicts_dict = get_count_dicts_from_df(mutation_count_df, context_count_dict, min_mer, max_mer)

    ## Now save each of the count dicts 
    config_file = "{}/1_{}mer.{}.{}.hardcoded_count_files.yaml".format(output_dir, max_mer, pop, feature)
    
    config = open(config_file, 'w')
    
    config.write("### yaml file holding count files for data.\n")
    config.write("---\n{}:\n  {}:\n".format(pop, feature))

    
    for mer in range(min_mer, max_mer + 1):
        mer_string = str(mer) + "mer"
        
        config.write("    {}:\n".format(mer_string))
        
        count_dict = count_dicts_dict[mer_string]
        output_file = "{}/{}.{}.{}.{}.folded_count_dict.json".format(output_dir, mer_string, dataset, pop, feature)
        with open(output_file, 'w') as jFile:
            json.dump(count_dict, jFile)

        config.write("      {}: {}\n".format(dataset, output_file))
        
    config.write('...')

def prepare_count_json_from_csv(context_count_file, dataset):
    
    # generate the dataframe column names
    names_list = ["Context"]    
    for chrom in range(1, 23):
        odd_chrom_name = str(chrom) + ".odd_bp"
        even_chrom_name = str(chrom) + ".even_bp"

        if dataset == "ALL" or dataset == "ODD":
            names_list.append(odd_chrom_name)
        
        if dataset == "ALL" or dataset == "EVEN":
            names_list.append(even_chrom_name)
   
    gw_full_df = pd.read_csv(context_count_file, sep='\t')
    dataset_df = gw_full_df[["Context"]]
    dataset_df["Count"] = gw_full_df[names_list].sum(axis=1)

    dataset_dict = dataset_df.set_index()["Count"]
    
    return dataset_dict 

def get_total_matching_muts(count_dict):
    
    total_muts = 0
    for context in count_dict:
        total_contexts = count_dict[context][4]
        total_ref_positions = count_dict[context][count_dict[context][5]]
        total_muts += total_contexts - total_ref_positions
 
    return total_muts

def get_context_index_dict(context_array):
    
    context_index_dict = {}
    i = 0
    for context in context_array:
        
        try:
            context_index_dict[context].append(i)
        except KeyError:
            context_index_dict[context] = [i]

        i += 1

    return context_index_dict

def get_count_dicts_from_df(df, raw_count_dict, min_mer, max_mer):
    
    context_ref_index = int(max_mer / 2)
    if max_mer % 2 == 0:
        context_ref_index -= 1


    max_count_dict = tabulate_max_window_counts_from_df(df, raw_count_dict, context_ref_index)
    ## TESTING
    total_count = 0
    n = 0
    print('max count dict generated')
    count_dicts_dict = simulation_functions.extrapolate_context_counts_down_tree(max_count_dict, min_mer, max_mer)
    return count_dicts_dict

def tabulate_max_window_counts_from_df(df, raw_count_dict, context_ref_index):

    max_count_dict = {}

    mutation_index_dict = {"A": 0, "C": 1, "G": 2, "T": 3}

    grouped_df = df.groupby(['Context', 'Mutation']).size().reset_index(name='counts')
    
    for idx, row in grouped_df.iterrows():

        context = row['Context']
        try:
            count_entry = max_count_dict[context]
            ref_pos = count_entry[5]
        except KeyError:
            count_entry = [0, 0, 0, 0] + raw_count_dict[context] + [0, 0]

            max_count_dict[context] = count_entry
            ref_index = mutation_index_dict[context[context_ref_index]]
            count_entry[5] = ref_index
            count_entry[6] = context_ref_index
            count_count_entry[5] = count_entry[4]
        
        mutation = row['Mutation']
        count = row['counts']

        mutation_index = mutation_index_dict[mutation]
        count_entry[mutation_index] = count
        count_entry[ref_index] = count_entry[ref_index] - count
    
    add_missing_contexts(max_count_dict, raw_count_dict)
    
    return max_count_dict

def add_missing_contexts(max_count_dict, raw_count_dict):
    
    missing_contexts = set(raw_count_dict.keys()) - set(max_count_dict.keys())
    print("number of missing contexts: ", len(missing_contexts))
    for context in missing_contexts:
        count_entry = raw_count_dict[context]
        total_contexts = count_entry[4]
        empty_count_entry = [0, 0, 0, 0] + count_entry[4:]
        #print(empty_count_entry)
        # set ref value to the total context count
        empty_count_entry[count_entry[5]] = total_contexts

        max_count_dict[context] = empty_count_entry


def parse_histogram_bins(hist_bins_file):
    
    bin_dict = {}
    bins_list = []
    
    hist_bins_df = pd.read_csv(hist_bins_file, index_col = False)
    
    bin_dict = dict(zip(hist_bins_df['breaks'], hist_bins_df['counts']))

    bin_list = list(hist_bins_df['breaks'])

    return (bin_dict, bin_list)


#######################################################################################################################################################
#cProfile.run('main(sys.argv[1:])')
         
if __name__ == "__main__":
    main(sys.argv[1:])
