#!/usr/bin/Python

# Created by: Christopher J Adams 5/26/2020
'''
 This script will read through fasta files and count all required info
 for count dicts

'''

#import cProfile
import sys
import os
import getopt
from Bio import SeqIO
import json
from itertools import product
import pandas as pd
import multiprocessing as mp
import gzip
import yaml

def help(error_num=1):
    print("""-----------------------------------------------------------------
ARGUMENTS
    -c => <yaml> config file REQUIRED
    --feature => <string> feature of interest REQUIRED
    -m => <int> mer length REQUIRED
    --co => <string> Specifies context count output file REQUIRED
    -b => <int> buffer shift of the given fasta OPTIONAL
          Default: same as the mer-length symmetric flank
    -a => <int> if asymmetric, designates the offset from the center nucleotide
          that is allowed OPTIONAL Default: symmetric mers, (i.e -a 0)
    -u => <boolean> specifies that you want unfolded contexts OPTIONAL
    -h => <boolean> specifies you only want high-confidence bases OPTIONAL
ASSUMPTIONS
    * fasta files only contain regions of interest
    * if the mer length is not odd, then default behavior is for the mutated nuc
      to be shifted down one. e.g 6mer, offset = 0, => NNN*NNN
    * for odd mers, if -a flag is used, the offset should not exceed 
      the mer flank length in either direction
    * Positive and negative offset values move the central nucleotide
      closer to the end and start of the fasta, respectively
      e.g 5mer, 0 => NN*NN ; 5mer, +1 => NNN*N ; 5mer, -1=>  N*NNN
    * Assumes that if there is a buffer, then it's being used in the pipeline and 
      you want to make sure that the entire window of possible asymmetries is included.
      I.e buffer of 10 corresponds to an asymmetric 11mer. Thus the full window to examine
      would be 21 total bp, surrounding the central nucleotide
NOTES
""")
    sys.exit(error_num)

###############################################################################

## MAIN 

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "c:p:m:o:b:a:u:h", ['feature=', 'co='])
    except getopt.GetoptError:
        print("Error: Incorrect usage of getopts flags!")
        help() 

    options_dict = dict(opts)

    ## Required arguments
    try:
        config_file = options_dict['-c']
        feature = options_dict['--feature']
        mer_length = options_dict['-m']
        context_output_file = options_dict['--co']
 
    except KeyError:
        print("Error: One of your required arguments does not exist.")
        help()
    
    ## Optional arguments
    buffer_bp = options_dict.get('-b', 0)
    offset = options_dict.get('-a', 0)
    unfolded = options_dict.get('-u', False)
    high_confidence = options_dict.get('-h', False)
    if high_confidence == '':
        high_confidence = True
    check_arguments(mer_length, offset, buffer_bp)

    print("Acceptable Inputs Given")

    driver(config_file = config_file, feature = feature, mer_length = int(mer_length), context_output_file = context_output_file, offset = int(offset), buffer_bp = int(buffer_bp), unfolded = unfolded, high_confidence = high_confidence)


## Makes sure that all the arguments given are congruent with one another.
## ONE-TIME CALL -- called by main

def check_arguments(mer_length, offset, buffer_bp):  
    
    for in_value in [mer_length, offset, buffer_bp]:
        try:
            value = int(in_value)
        except ValueError:
            print("Error: Unexpected non-integer value for input")
            help()
    
    mer_length = int(mer_length)
    offset = int(offset)
    buffer_bp = int(buffer_bp)

    # check that the offset does not overshoot the mer length
    if offset != 0:
        flank = int(mer_length / 2)
        if abs(offset) > flank:
            print("Error: offset overshoots the mer length")
            help()
    
    # when default buffer_bp is given, it is implied that the whole genome is being used
    if buffer_bp != 0:
        even_adj = 0
        if mer_length % 2 == 0:
            even_adj = 1
        if (int(mer_length / 2) + abs(offset) - even_adj) > buffer_bp:
            print("Total length of area necessary: {}".format(int(mer_length / 2) + abs(offset) - even_adj))
            print("Buffer area given: {}".format(buffer_bp))
            print("Error: specified mer overshoots the buffer region")
            help()


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(config_file, feature, mer_length, context_output_file, offset, buffer_bp, unfolded, high_confidence):
    
    #### GATHER/INIT GENERAL INFORMATION ####
    config_dict = yaml.load(open(config_file, 'r'), Loader=yaml.SafeLoader)
 
    fasta_file_dict = config_dict["features"][feature]['fastas']
    chrom_list = list(fasta_file_dict.keys()) 
    #### BEGIN PARALLELIZED CHROMOSOME COUNTS ####
    pool = mp.Pool(min([len(chrom_list), mp.cpu_count()]))
    chrom_results = [pool.apply(count_contexts, args=(chrom, fasta_file_dict, mer_length, offset, buffer_bp, unfolded, high_confidence)) for chrom in chrom_list]
    pool.close()
    
    #### COMBINE DICTIONARIES TOGETHER ####
    context_count_master_df = None
    first_it = True
    for chrom_context_count_dict, chrom_string in zip(chrom_results,chrom_list):
        chrom_context_count_df = pd.DataFrame.from_dict(chrom_context_count_dict,
                                                        columns = [chrom_string + ".odd_bp", chrom_string + '.even_bp'],
                                                        orient = 'index')
        
        if first_it:
            context_count_master_df = chrom_context_count_df
            first_it = False
        else:
            ## merge
            context_count_master_df = context_count_master_df.join(chrom_context_count_df, how='inner')
    
    context_count_master_df["Context"] = context_count_master_df.index
    context_count_master_df.to_csv(context_output_file, sep = '\t', index = False)

    print("Output file successfully saved")

def count_contexts(chrom, fasta_file_dict, mer_length, offset, buffer_bp, unfolded, high_confidence):

    #### GATHER/INIT GENERAL INFORMATION ####
    full_region_length = mer_length
    if buffer_bp:   
        full_region_length = buffer_bp * 2 + 1
    
    mut_nuc_pos = int(full_region_length / 2)
     
    odd_adjustment = 0
    if float(mer_length) % 2.0 == 0.0 and offset > 0:
        odd_adjustment = 1
 
    flank = int(mer_length / 2)
    left_flank = flank + offset - odd_adjustment 
    right_flank = mer_length - left_flank
    left_seq_edge = buffer_bp-left_flank
    right_seq_edge =  buffer_bp-right_flank
 
    metadata_dict = {'n': 0, 'nuc_n': 0, 'total_regions': 0}

    context_count_dict = initialize_count_dicts(mer_length, left_flank, right_flank, unfolded)
    
    fasta_file = fasta_file_dict[chrom]
    

    #### BEGIN READING THROUGH FASTA ####
    # open the fasta and check whether it's zipped
    if fasta_file[-2:] == "gz":
        with gzip.open(fasta_file, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                fasta_seq = str(record.seq)
                fasta_pos = int(str(record.id).strip().split(':')[1].split('-')[0]) + mut_nuc_pos + 1
                iterate_through_fasta_seq(context_count_dict, fasta_seq, fasta_pos, mer_length, offset, mut_nuc_pos, full_region_length, unfolded, left_seq_edge, high_confidence)
             
    else:
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                fasta_seq = str(record.seq)
                fasta_pos = int(str(record.id).strip().split(':')[1].split('-')[0]) + mut_nuc_pos + 1
                iterate_through_fasta_seq(context_count_dict, fasta_seq, fasta_pos, mer_length, offset, mut_nuc_pos, full_region_length, unfolded, left_seq_edge, high_confidence)
    
    return context_count_dict


## iterate through fasta to generate counts dict
## ONE-TIME CALL -- called by driver

def iterate_through_fasta_seq(context_count_dict, fasta_seq, fasta_pos, mer_length, offset, mut_nuc_pos, full_region_length, unfolded, left_seq_edge, high_confidence):
    
    complement_dict = {"A":"T", "T":"A", "G":"C", "C":"G"}
    
    current_region = fasta_seq[0:full_region_length]

    add_mer_to_dict(current_region, mer_length, mut_nuc_pos, context_count_dict, complement_dict, offset, fasta_pos, fasta_seq, unfolded, full_region_length, left_seq_edge, high_confidence)
    fasta_pos += 1
 
    for nuc in fasta_seq[full_region_length:]:
        #current_region = buffer(current_region, 1, shift)[:] + nuc
        current_region = current_region[1:] + nuc
        add_mer_to_dict(current_region, mer_length, mut_nuc_pos, context_count_dict, complement_dict, offset, fasta_pos, fasta_seq, unfolded, full_region_length, left_seq_edge, high_confidence)
        fasta_pos += 1
    #print("total # seq invalidated by Ns: {}\ntotal # Ns in fasta: {}".format(total_ns['n'],total_ns["nuc_n"]))

## Attempt to add mer to dict 
## MULTIPLE CALLS -- called by iterate_through_fasta_seq

def add_mer_to_dict(current_region, mer_length, mut_nuc_pos, context_count_dict, complement_dict, offset, fasta_pos, fasta_seq, unfolded, full_region_length, left_seq_edge, high_confidence):
   
    valid_mer = check_valid_mer(current_region, high_confidence)
    # if the mer is not valid, just return and continue to the next position
    if not valid_mer:
        return

    middle_nucs = 'CA'
    if unfolded:
        middle_nucs = 'ACGT'
   
    current_mer = current_region[left_seq_edge:left_seq_edge + mer_length].upper()
 
    if current_region[mut_nuc_pos].upper() not in middle_nucs:
        current_mer = get_reverse_comp_mer(current_mer, complement_dict)
 
    even_odd_index = fasta_pos % 2
    
    context_count_dict[current_mer][even_odd_index] += 1 


## find the reverse complementary mer
## MULTIPLE CALLS -- called by add_mer_to_dict

def get_reverse_comp_mer(mer, complement_dict):
    
    reverse_seq_list = [complement_dict[nuc] for nuc in mer[::-1]]
    reverse_seq = ''.join(reverse_seq_list)

    return reverse_seq


## Check if the mer is valid
## MULTIPLE CALLS -- called by add_mer_to_dict

def check_valid_mer(current_mer, high_confidence, mer_length = False):
    
    # Check that mer only contains acceptable mers
    good_nucs = ["A", "G", "C", "T", 'a', 'c', 'g' ,'t']
    if high_confidence:
        good_nucs = ["A", "G", "C", "T"]
    for nuc in current_mer:
        if nuc not in good_nucs:
            return False
    
    if mer_length:
        if len(current_mer) != mer_length:
            return False

    return True

## initialize mer_count_dict
## ONE-TIME CALL -- called by driver

def initialize_count_dicts(mer_length, left_flank, right_flank, unfolded):
    
    context_count_dict = {}

    middle_nuc = ["C", "A"]
    if unfolded:
        middle_nuc = ["A", "C", "G", "T"]
    
    for combination in product('ACGT', repeat=mer_length):
        context = ''.join(combination)
        # make sure none of the reverse contexts make it into the counts dict
        if context[left_flank] not in middle_nuc:
            continue
        else:
            context_count_dict[context] = [0, 0]

    
    return context_count_dict

    
#######################################################################################################################################################
#cProfile.run('main(sys.argv[1:])')
         
if __name__ == "__main__":
    main(sys.argv[1:])
