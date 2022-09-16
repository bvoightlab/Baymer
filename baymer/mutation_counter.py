#!/usr/bin/Python

# Created by: Christopher J Adams 1/19/2022
'''
 This script will read through fasta and vcf files and count all required info
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
import concurrent.futures
import gzip
import yaml

def help(error_num=1):
    print("""-----------------------------------------------------------------
ARGUMENTS
    -c => <yaml> vcf config file REQUIRED
    --feature => <string> feature of interest REQUIRED
    -p => <string> population of interest REQUIRED
    -m => <int> mer length REQUIRED
    --mo => <string> Specifies mutation count output file REQUIRED
    --ac => <int> specifies the exact allele count in the dataset that you would like to keep REQUIRED
    --min => <bool> specifies that you want *at least* X number of AC
    -b => <int> buffer shift of the given fasta OPTIONAL
          Default: same as the mer-length symmetric flank
    -a => <int> if asymmetric, designates the offset from the center nucleotide
          that is allowed OPTIONAL Default: symmetric mers, (i.e -a 0)
    -u => <boolean> specifies that you want unfolded contexts OPTIONAL
    -h => <boolean> specifies you only want high-confidence bases OPTIONAL
    --nygc => <bool> specifies the vcfs belong to nygc variants OPTIONAL Default: assumes gnomad
    --fasta-consistent => <bool> specifies that you would like enforce vcf consistency (e.g fasta is ancestral but vcf is not) OPTIONAL
ASSUMPTIONS
    * fasta files only contain regions of interest
    * if the mer length is not odd, then default behavior is for the mutated nuc
      to be shifted up one. e.g 6mer, offset = 0, => NNN*NN
    * for odd mers, if -a flag is used, the offset should not exceed 
      the mer flank length in either direction
    * Positive and negative offset values move the central nucleotide
      closer to the end and start of the fasta, respectively
      e.g 5mer, 0 => NN*NN ; 5mer, +1 => NNN*N ; 5mer, -1=>  N*NNN
    * Assumes that if there is a buffer, then it's being used in the pipeline and 
      you want to make sure that the entire window of possible asymmetries is included.
      I.e buffer of 10 corresponds to an asymmetric 11mer. Thus the full window to examine
      would be 21 total bp, surrounding the central nucleotide
""")
    sys.exit(error_num)

###############################################################################

## MAIN 

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "c:p:m:o:b:a:u:h", ['feature=', 'mo=', 'ac=', 'nygc', 'min', "fasta-consistent"])
    except getopt.GetoptError:
        print("Error: Incorrect usage of getopts flags!")
        help() 

    options_dict = dict(opts)

    ## Required arguments
    try:
        config_file = options_dict['-c']
        feature = options_dict['--feature']
        pop = options_dict['-p']
        mer_length = options_dict['-m']
        mutation_output_file = options_dict['--mo']
        allele_count = int(options_dict['--ac'])

    except KeyError:
        print("Error: One of your required arguments does not exist.")
        help()
    
    ## Optional arguments
    buffer_bp = options_dict.get('-b', 0)
    offset = options_dict.get('-a', 0)
    unfolded = options_dict.get('-u', False)
    nygc_bool = options_dict.get('--nygc', False)
    high_confidence = options_dict.get('-h', False)
    min_bool = options_dict.get('--min', False)
    fasta_consistent = options_dict.get('--fasta-consistent', False)
    
    if high_confidence == '':
        high_confidence = True

    if min_bool == '':
        min_bool = True

    if fasta_consistent == "":
        fasta_consistent = True
    
    if nygc_bool == "":
        nygc_bool = True

    check_arguments(mer_length, offset, buffer_bp)

    print("Acceptable Inputs Given")

    driver(config_file, pop, feature, int(mer_length), mutation_output_file, int(offset), int(buffer_bp), unfolded, high_confidence, allele_count, nygc_bool, min_bool, fasta_consistent)


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
        if (int(mer_length / 2) + abs(offset)) > buffer_bp:
            print("Total length of area necessary: {}".format(int(mer_length / 2) + abs(offset)))
            print("Buffer area given: {}".format(buffer_bp))
            print("Error: specified mer overshoots the buffer region")
            help()


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(config_file, pop, feature, mer_length, mutation_output_file, offset, buffer_bp, unfolded, high_confidence, allele_count, nygc_bool, min_bool, fasta_consistent):
    
    #### GATHER/INIT GENERAL INFORMATION ####
    config_dict = yaml.load(open(config_file, 'r'), Loader=yaml.SafeLoader)
 
    chrom_list = config_dict["chromosomes"]
    fasta_file_dict = config_dict['features'][feature]['fastas']
    vcf_file_dict = config_dict[pop]['vcf_files']
    quality_filter = "VQSLOD"
    if not nygc_bool:
        quality_filter = "AS_VQSLOD"


    ac_info = ['AC_' + pop, 'AN_' + pop, allele_count, min_bool, quality_filter]
    if nygc_bool or pop == "all_pops":
        ac_info = ['AC', 'AN', allele_count, min_bool, quality_filter]
    
    #### BEGIN PARALLELIZED CHROMOSOME COUNTS ####
    chrom_results = None
    with concurrent.futures.ProcessPoolExecutor() as executor:
        chrom_results = [executor.submit(count_mutations, chrom, vcf_file_dict, fasta_file_dict, mer_length, offset, buffer_bp, unfolded, high_confidence, fasta_consistent, ac_info) for chrom in chrom_list]
    
    first_it = True
    master_df = None
    df_columns = ["Chrom","Position","Context","Mutation","AC","AN","quality_score", "even_odd_bool"]
     
    for r in concurrent.futures.as_completed(chrom_results):
        context_mutation_list = r.result()
        context_mutation_df = pd.DataFrame(data = context_mutation_list, columns = df_columns)
        if first_it:
            master_df = context_mutation_df
            first_it = False
        else:
            master_df = master_df.append(context_mutation_df)

    print("Total size of df before qc: ", len(master_df))
    # remove any multiallelic sites
    #master_df.drop_duplicates(subset = ["Chrom", "Position"], keep = False, inplace = True)
    #print("Total size of df after removing all multiallelic sites: ", len(master_df))
    master_df.to_csv(mutation_output_file, index = False)
    
    '''
    pool = mp.Pool(len(chrom_list))
    chrom_results = [pool.apply(count_mutations, args=(chrom, vcf_file_dict, fasta_file_dict, mer_length, offset, buffer_bp, unfolded, high_confidence, fasta_consistent, ac_info)) for chrom in chrom_list]
    pool.close()
    #### TESTING
    #chrom_results = count_mutations('chr22', vcf_file_dict, fasta_file_dict, mer_length, offset, buffer_bp, unfolded, high_confidence, af_info)
    #### TESTING
    
    #### COMBINE DICTIONARIES TOGETHER ####
    first_it = True
    out_file = open(mutation_output_file, 'w')
    for context_mutation_list in chrom_results:
        if first_it:
            out_file.write("Chrom,Position,Context,Mutation,AC,AN,quality_score,even_odd_bool\n")
            first_it = False
        else:
            master_df = master_df.append(context_mutation_df)
    #'''
    print("Output file successfully saved")
    print(mutation_output_file)

def count_mutations(chrom, vcf_file_dict, fasta_file_dict, mer_length, offset, buffer_bp, unfolded, high_confidence, fasta_consistent, ac_info):
    
    print(chrom)
    fasta_qc_list = [0, 0, 0]
    #### GATHER/INIT GENERAL INFORMATION ####
    middle_nucs = 'CA'
    if unfolded:
        middle_nucs = 'ACGT'

    odd_adjustment = 0
    if float(mer_length) % 2.0 == 0.0 and offset > 0:
        odd_adjustment = 1
 
    flank = int(mer_length / 2)
    left_flank = flank + offset - odd_adjustment 
    right_flank = mer_length - left_flank
    left_seq_edge = buffer_bp-left_flank
    right_seq_edge =  buffer_bp-right_flank
 
    #mut_count_dict = initialize_mut_count_dict(left_flank, right_flank, unfolded)
    fasta_file = fasta_file_dict[chrom]
    vcf_file = vcf_file_dict[chrom]

    ## open vcf file and find first instance of line corresponding to a variant of interest ##
    vcf = gzip.open(vcf_file, 'r')
    header_line = True
    while header_line:
        line = str(vcf.readline(), 'utf-8')
        if not line.startswith('##'):
            header_line = False        
    relevant_vcf_list = get_next_appropriate_line(vcf, ac_info, fasta_pos = False)
    
    context_mutation_list = []

    # tracks the position from the start of the fasta string corresponding to the mut
    full_region_length = mer_length
    if buffer_bp:
        #print("buffer bp: ", buffer_bp)
        full_region_length = buffer_bp * 2 + 1
        #print(full_region_length)
        
    mut_nuc_pos = int(full_region_length / 2)
    #### BEGIN READING THROUGH FASTA ####
    with open(fasta_file, 'r') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            fasta_seq = str(record.seq)
            fasta_start_pos = int(str(record.id).strip().split(':')[1].split('-')[0]) + mut_nuc_pos + 1
            fasta_seq_end_pos = int(str(record.id).strip().split(':')[1].split('-')[1]) - mut_nuc_pos + 1
            # check whether there are mutations within the interval given
            # make sure the vcf line is in the appropriate position
            if fasta_start_pos > relevant_vcf_list[0]:
                relevant_vcf_list = get_next_appropriate_line(vcf, ac_info, fasta_pos = fasta_start_pos)
                # if we have reached the end of the vcf
                if not relevant_vcf_list:
                    break
                
            #elif fasta_seq_end_pos < relevant_vcf_list[0]:
                #continue
            mutations_within_interval = bool(fasta_seq_end_pos >= relevant_vcf_list[0])
            
            
            while mutations_within_interval:
                mut_pos = int(relevant_vcf_list[0])
                adj_mut_pos = mut_pos - (fasta_start_pos - mut_nuc_pos)
                # check for bug
                if adj_mut_pos < mut_nuc_pos:
                    print(adj_mut_pos)
                    print("Error: mutation in buffer region")
                    help()

                start_interval = adj_mut_pos - mut_nuc_pos
                end_interval = start_interval + full_region_length
                current_region = fasta_seq[start_interval:end_interval]

                valid_mer = check_valid_mer(current_region, high_confidence, full_region_length)
                if not valid_mer:
                    relevant_vcf_list = get_next_appropriate_line(vcf, ac_info, mut_pos)
                    # if we have reached the end of the vcf (relevant_vcf_list only equals False if it's the final mut)
                    if not relevant_vcf_list:
                        break
                    mutations_within_interval = bool(fasta_seq_end_pos >= relevant_vcf_list[0])
                    continue
                    
                current_mer = current_region[left_seq_edge:left_seq_edge + mer_length].upper()
                if fasta_consistent:
                    fasta_ref = current_region[mut_nuc_pos]
                    fasta_compatible_line = convert_relevant_vcf_list(relevant_vcf_list, current_mer, fasta_ref, ac_info, fasta_qc_list)
                    if not fasta_compatible_line:
                        # go to the next vcf line
                        relevant_vcf_list = get_next_appropriate_line(vcf, ac_info, mut_pos)
                        # if we have reached the end of the vcf (relevant_vcf_list only equals False if it's the final mut)
                        if not relevant_vcf_list:
                            break
                        mutations_within_interval = bool(fasta_seq_end_pos >= relevant_vcf_list[0])
                        continue

                # make sure the reference alleles are identical
                if current_region[mut_nuc_pos].upper() != relevant_vcf_list[1]:
                    print("Error: reference nucleotide is off")
                    print(current_region[mut_nuc_pos].upper())
                    print(relevant_vcf_list[1])
                    print('--------')
                    help()
                if current_region[mut_nuc_pos].upper() not in middle_nucs:
                    current_mer = get_reverse_comp_mer(current_mer)
                    relevant_vcf_list[2] = get_reverse_comp_mer(relevant_vcf_list[2])
                even_odd_index = mut_pos % 2
                mutation_entry = [relevant_vcf_list[6], mut_pos, current_mer, relevant_vcf_list[2], float(relevant_vcf_list[3]), float(relevant_vcf_list[4]), float(relevant_vcf_list[5]), even_odd_index] 
                context_mutation_list.append(mutation_entry)
                relevant_vcf_list = get_next_appropriate_line(vcf, ac_info, mut_pos)
                if not relevant_vcf_list:
                    mutations_within_interval = False
                else:
                    mutations_within_interval = bool(fasta_seq_end_pos >= relevant_vcf_list[0])
            
            if not relevant_vcf_list:
                break

    ## close files ##
    vcf.close()
    
    print("VCF REF matched ancestral: ", fasta_qc_list[0])
    print("VCF ALT matched ancestral: ", fasta_qc_list[1])
    print("Could not find ancestral match: ", fasta_qc_list[2])

    return context_mutation_list


def convert_relevant_vcf_list(relevant_vcf_list, current_mer, fasta_ref, ac_info, fasta_qc_list):
    
    fasta_compatible_line = False
    vcf_ref = str(relevant_vcf_list[1])
    vcf_alt = str(relevant_vcf_list[2])
    if fasta_ref == vcf_ref:
        fasta_compatible_line = True
        fasta_qc_list[0] += 1
    elif fasta_ref != vcf_ref:
        if fasta_ref == vcf_alt:
            fasta_qc_list[1] += 1
            ## swap gts
            # exchange alt allele (vcf ref becomes vcf alt)
            relevant_vcf_list[2] = vcf_ref
            # exchange ref allele
            relevant_vcf_list[1] = fasta_ref
        
            # recalculate AC
            new_ac = relevant_vcf_list[4] - relevant_vcf_list[3]
            min_bool = ac_info[3]

            relevant_vcf_list[3] 

            if not min_bool:
                if new_ac == ac_info[2]:
                    fasta_compatible_line = True
            else:
                if new_ac >= ac_info[2]:
                    fasta_compatible_line = True
            
            relevant_vcf_list[3] = new_ac
        else:
            fasta_qc_list[2] += 1 
    return fasta_compatible_line

def get_next_appropriate_line(vcf, ac_info, fasta_pos = False):
    last_mut = False
    appropriate_line = False
    while not appropriate_line:
        line = str(vcf.readline(), 'utf-8')
        ## need a check for what happens if the file ends
        line_list = line.strip().split('\t')
        try:
            line_list[1]
        except IndexError:
            last_mut = True
            break
        if fasta_pos and not last_mut:
            if int(line_list[1]) < fasta_pos:
                continue
        appropriate_line = check_line_status(line_list, ac_info)
    
    if last_mut:
        return False
    else:
        # subtract one to zero index the coordinates
        return [int(line_list[1]), line_list[3], line_list[4], appropriate_line[0], appropriate_line[1], appropriate_line[2], int(line_list[0][3:])]

def check_line_status(line_list, ac_info):
    
    #check for multiallelic lines
    if len(line_list[3]) + len(line_list[4]) > 2:
        return False

    info_col = line_list[7]
    info_line_list = info_col.strip().split(';')
    
    features_found = 0
    an = 0
    ac = 0
    quality_score = None
    min_bool = ac_info[3]
    quality_string = ac_info[4]
    annos_found = 0
    for info_anno in info_line_list:
        split_info_anno = info_anno.strip().split('=')
        feature = str(split_info_anno[0])
        # check that min af count is large enough if applicable
        if feature == ac_info[0]:
            ac = int(split_info_anno[1])
            features_found += 1
            if not min_bool:
                if ac != ac_info[2]:
                    return False
            else:
                if ac < ac_info[2]:
                    return False

            annos_found += 1
            if annos_found == 3:
                break
        elif feature == ac_info[1]:    
            an = int(split_info_anno[1])
            annos_found += 1
            if annos_found == 3:
                break
        elif feature == quality_string:
            quality_score = float(split_info_anno[1])
            annos_found += 1
            if annos_found == 3:
                break
    if annos_found != 3:
        return False
    
    return (ac, an, quality_score)


## find the reverse complementary mer
## MULTIPLE CALLS -- called by add_mer_to_dict

def get_reverse_comp_mer(mer):
    
    complement_dict = {"A":"T", "T":"A", "G":"C", "C":"G"}
 
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

    
#######################################################################################################################################################
#cProfile.run('main(sys.argv[1:])')
         
if __name__ == "__main__":
    main(sys.argv[1:])
