#!/usr/bin/Python

# Created by: Christopher J Adams 9/8/2022
# 

###############################################################################
###
### This script will convert a variant list into a count dict given certain ac specifications
###
###############################################################################


## NOTE Assumes symmetric-ish counts

#import cProfile
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

def help(exit_num=1):
    print("""-----------------------------------------------------------------
ARGUMENTS
    -c => <yaml> config file REQUIRED
    --mc => <csv> mutations csv REQUIRED
    --cc => <json> contexts csv REQUIRED
    -o => <dir> output dir REQUIRED
    -p => <str> pop REQUIRED
    -f => <str> feature REQUIRED
    -d => <str> dataset REQUIRED
    --max-af => <float> maximum allele frequency which is considered OPTIONAL Default: 0.85
    --min-af => <float> minimum allele frequency which is considered OPTIONAL Default: 0
    --min-ac => <int> minimum allele count to include OPTIONAL Default 1

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
        opts, args = getopt.getopt(sys.argv[1:], "c:o:p:f:d:", ["max-af=", "min-af=", "quality=", "cc=", "mc=", "min-ac="])
                                                              
    except getopt.GetoptError:
        print("Error: Incorrect usage of getopts flags!")
        help()

    options_dict = dict(opts)
    
    ## Required arguments
    try:
        config_file = options_dict['-c']
        output_dir = options_dict['-o']
        mutation_count_file = options_dict['--mc']
        context_count_file = options_dict['--cc']
        feature = options_dict['-f']
        pop = options_dict['-p']
        dataset = options_dict['-d']
    
    except KeyError:
        print("Error: One of your required arguments does not exist.")
        help()
    
    # optional arguments
    max_af = float(options_dict.get("--max-af", 0.85))
    min_af = float(options_dict.get("--min-af", 0.0))
    quality = options_dict.get("--quality", False)
    min_ac = int(options_dict.get("--min-ac", 1))
    print("Acceptable Inputs Given")

    driver(config_file, mutation_count_file, context_count_file, feature, pop, dataset, output_dir, max_af, min_af, quality, min_ac)


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(config_file, mutation_count_file, context_count_file, feature, pop, dataset, output_dir, max_af=0.85, min_af = 0, quality=False, min_ac = 1):
    
    config_dict = yaml.load(open(config_file, 'r'), Loader=yaml.SafeLoader)
    
    # prepare the master count dict from the context count file
    context_count_dict = prepare_count_json_from_csv(context_count_file, dataset, config_dict)
    

    # mutation count data
    mutation_count_df = pd.read_csv(mutation_count_file)
    if dataset == "EVEN":
        mutation_count_df = mutation_count_df.loc[mutation_count_df["even_odd_bool"] == 0]
    elif dataset == "ODD":
        mutation_count_df = mutation_count_df.loc[mutation_count_df["even_odd_bool"] == 1]
    
    mutation_count_df['AF'] = mutation_count_df['AC'] / mutation_count_df['AN']
    mutation_count_df = mutation_count_df.loc[(mutation_count_df["AF"] < max_af) & (mutation_count_df["AF"] >= min_af)]
    mutation_count_df = mutation_count_df.loc[mutation_count_df["AC"] >= min_ac]
    print(mutation_count_df.head()) 
    if quality is not False:
        mutation_count_df = mutation_count_df.loc[mutation_count_df["quality_score"] >= float(quality)]
    total_muts = mutation_count_df.count()[0]
    print("variants counted from mutation file: ", total_muts)
    print("mean ac: ", np.mean(mutation_count_df["AC"]))
    print("median ac: ", np.median(mutation_count_df["AC"]))
    print("min ac: ", np.min(mutation_count_df["AC"])) 
    print("max ac: ", np.max(mutation_count_df["AC"])) 
    print("mean af: ", np.mean(mutation_count_df["AF"]))
    min_mer = 1
    max_mer = len(list(context_count_dict.keys())[0])
    
    count_dicts_dict = get_count_dicts_from_df(mutation_count_df, context_count_dict, min_mer, max_mer)

    ## Now save each of the count dicts 
    config_file = "{}/1_{}mer.{}.{}.{}.hardcoded_count_files.yaml".format(output_dir, max_mer, dataset, pop, feature)
    
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


def prepare_count_json_from_csv(context_count_file, dataset, config_dict):
    
    # generate the dataframe column names
    names_list = ["Context"]    
    for chrom in config_dict["chromosomes"]:
        odd_chrom_name = str(chrom) + ".odd_bp"
        even_chrom_name = str(chrom) + ".even_bp"

        if dataset == "ALL" or dataset == "ODD":
            names_list.append(odd_chrom_name)
        
        if dataset == "ALL" or dataset == "EVEN":
            names_list.append(even_chrom_name)
   
    gw_full_df = pd.read_csv(context_count_file, sep='\t')
    dataset_df = gw_full_df[["Context"]]
    dataset_df["Count"] = gw_full_df[names_list].sum(axis=1)
    dataset_dict = dict(zip(dataset_df["Context"], dataset_df["Count"]))
    
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

    folded_raw_count_dict = fold_raw_count_dict(raw_count_dict, context_ref_index)

    max_count_dict = tabulate_max_window_counts_from_df(df, folded_raw_count_dict, context_ref_index)
    ## TESTING
    total_count = 0
    n = 0
    print('max count dict generated')
    count_dicts_dict = extrapolate_context_counts_down_tree(max_count_dict, min_mer, max_mer)
    return count_dicts_dict

def fold_raw_count_dict(raw_count_dict, context_ref_index):
    
    folded_count_dict = {}
    for c in raw_count_dict:
        total_contexts = raw_count_dict[c]
        

        if c[context_ref_index] not in "AC":
            c = get_reverse_comp_mer(c)
            
        if c in folded_count_dict:
            folded_count_dict[c] += total_contexts
        else:
            folded_count_dict[c] = total_contexts

    return folded_count_dict

def tabulate_max_window_counts_from_df(df, raw_count_dict, context_ref_index):

    max_count_dict = {}

    mutation_index_dict = {"A": 0, "C": 1, "G": 2, "T": 3}

    grouped_df = df.groupby(['Context', 'Mutation']).size().reset_index(name='counts')
    print(context_ref_index)
    for idx, row in grouped_df.iterrows():

        context = row['Context']
        total_contexts = raw_count_dict[context]
        try:
            ref_index = max_count_dict[context][5]

        except KeyError:
            ref_index = mutation_index_dict[context[context_ref_index]]
            max_count_dict[context] = [0, 0, 0, 0, total_contexts, ref_index, context_ref_index]
            max_count_dict[context][ref_index] = total_contexts 
        mutation = row['Mutation']
        count = row['counts']
        
        mutation_index = mutation_index_dict[mutation]
        max_count_dict[context][mutation_index] = count
        max_count_dict[context][ref_index] -= count
    add_missing_contexts(max_count_dict, raw_count_dict, context_ref_index)
    '''
    # fix multiallelic sites by adding an additional context to the data
    multiallelic_sites = df[df.duplicated(subset=["Chrom", "Position", "Context"], keep = False)]
    grouped_multiallelic_sites = multiallelic_sites.groupby(["Chrom", "Position", 'Context']).size().reset_index(name='counts')
    print(grouped_multiallelic_sites.head())
    print(len(grouped_multiallelic_sites))
    for idx, row in grouped_multiallelic_sites.iterrows():
        context = row["Context"]
        num_sites = row["counts"]
        sites_to_add = num_sites - 1
        max_count_dict[context][4] += sites_to_add
    '''
    qc_check_max_count_dict(max_count_dict)
    
    return max_count_dict

def get_reverse_comp_mer(mer):

    complement_dict = {"A":"T", "T":"A", "G":"C", "C":"G"}

    reverse_seq_list = [complement_dict[nuc] for nuc in mer[::-1]]
    reverse_seq = ''.join(reverse_seq_list)

    return reverse_seq

def qc_check_max_count_dict(max_count_dict):
    
    for c in max_count_dict:
        ref_index = max_count_dict[c][5]
        total_contexts = 0
        total_no_muts = 0
        for i in range(4):
            muts = max_count_dict[c][i]
            if muts < 0:
                if i != ref_index:
                    print("Error: this context has negative mutations")
                    print(c)
                    print(max_count_dict[c])
                    help()
                elif i == ref_index:
                    max_count_dict[c][i] = 0
            total_contexts += muts
            if i == ref_index:
                total_no_muts = muts

        if total_contexts != max_count_dict[c][4]:
            print("Error: miscounting in count dict")
            print(c)
            print(max_count_dict[c])
            print("total contexts counted: ", total_contexts)
            help()

def add_missing_contexts(max_count_dict, raw_count_dict, context_ref_index):
    
    mutation_index_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
    
    missing_contexts = set(raw_count_dict.keys()) - set(max_count_dict.keys())    
    print("number of missing contexts: ", len(missing_contexts))
    for context in missing_contexts:
        total_contexts = raw_count_dict[context]
        ref_index = mutation_index_dict[context[context_ref_index]]
        empty_count_entry = [0, 0, 0, 0, total_contexts, ref_index, context_ref_index]
        #print(empty_count_entry)
        # set ref value to the total context count
        empty_count_entry[ref_index] = total_contexts

        max_count_dict[context] = empty_count_entry
    

def parse_histogram_bins(hist_bins_file):
    
    bin_dict = {}
    bins_list = []
    
    hist_bins_df = pd.read_csv(hist_bins_file, index_col = False)
    
    bin_dict = dict(zip(hist_bins_df['breaks'], hist_bins_df['counts']))

    bin_list = list(hist_bins_df['breaks'])

    return (bin_dict, bin_list)

def extrapolate_context_counts_down_tree(max_count_dict, min_mer = 3, max_mer = 9, oppo_asymmetric = False):
    
    counts_dict = {}
    # set up counts dict
    for mer_size in range(min_mer, max_mer + 1):
        counts_dict[str(mer_size) + 'mer'] = {}
    total_count = 0
    n = 0
    for context in max_count_dict:
        context_list = max_count_dict[context]
        nuc_in_scope = context_list[5]
        
        samp_size = context_list[4]
        count_entry = context_list[:] 
        counts_dict[str(max_mer) + 'mer'][context] = count_entry
        # next extrapolate the counts down the tree
        smaller_mer = context
        temp_mer_size = max_mer - 1
        temp_ref_context_index = int(context_list[6])
        while temp_mer_size >= min_mer:
            temp_mer_size_string = str(temp_mer_size) + 'mer'
            
            # if context is odd
            if temp_mer_size % 2 == 1:
                if not oppo_asymmetric:
                    smaller_mer = smaller_mer[:-1]
                else:
                    smaller_mer = smaller_mer[1:]
                    temp_ref_context_index -= 1
            # if mer is even
            else:
                if not oppo_asymmetric:
                    smaller_mer = smaller_mer[1:]
                    temp_ref_context_index -= 1
                else:
                    smaller_mer = smaller_mer[:-1]
            
            try:
                for pos in range(4):
                    counts_dict[temp_mer_size_string][smaller_mer][pos] += count_entry[pos]
                counts_dict[temp_mer_size_string][smaller_mer][4] += samp_size
                    
            except KeyError:
                new_count_entry = count_entry[:]
                new_count_entry[6] = temp_ref_context_index
                counts_dict[temp_mer_size_string][smaller_mer] = new_count_entry
            temp_mer_size -= 1 
    
    return counts_dict


#######################################################################################################################################################
#cProfile.run('main(sys.argv[1:])')
         
if __name__ == "__main__":
    main(sys.argv[1:])
