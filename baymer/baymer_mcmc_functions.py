## Baymer helper functions
import sys
import numpy as np
import random
from itertools import product

### FUNCTIONS ###

def initialize_layer_tree_edge_list(layer, layer_size, max_mer, old_context_list, old_p_vec_array, sigmas, alpha_probs, set_p_vec_sample_array, leaf_count_dict, first_sample, random_seed, zero_init, oppo_asymmetry = False):
    
    if random_seed:
        np.random.seed(int(random_seed))
        random.seed(int(random_seed))
    ## Init all the returned arrays
    theta_array = np.zeros((layer_size, 3), dtype=np.float32)
    theta_probabilities_array = np.zeros((layer_size, 3))
    p_vec_array = np.ones((layer_size, 3), dtype=np.float32)
    indicator_array = np.zeros((layer_size, 3), dtype=np.int32)
    alpha_probabilities_array = np.zeros((layer_size, 3))
    context_list = [None for x in range(layer_size)]
    leaf_count_components = 4**(int(max_mer) - int(layer))
    leaf_count_array = np.ones((layer_size, leaf_count_components))
    
    spike_sigma, slab_sigma = sigmas
    spike_alpha_prob, slab_alpha_prob = alpha_probs 

    # check if this is the first layer
    if layer == 0:
        context_list = np.array(['A', 'C'])
        array_index = -1
        for context in context_list:
            array_index += 1
            init_theta = np.random.uniform(low = 0.0, high = 0.02, size = 3)
            init_p_vec = init_theta
                
            theta_array[array_index] = init_theta
            p_vec_array[array_index] = init_p_vec
            
            # get leaf_counts
            leaf_contexts = get_leaf_contexts(context, max_mer, oppo_asymmetry)
            leaf_counts = None
            if context == 'C':
                leaf_counts = get_adjusted_leaf_counts(leaf_contexts, leaf_count_dict).astype(np.int32)
            else:
                leaf_counts = np.array([np.array(leaf_count_dict[x][0:4]) for x in leaf_contexts]).flatten().astype(np.int32)
            
            leaf_count_array[array_index] = leaf_counts
            
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
            for nuc in ['A', 'C', 'G', 'T']:
                array_index = index_mod + (parent_index * 4)
                
                
                child_context = nuc + parent_context
                if (not oppo_asymmetry and odd_bool) or (oppo_asymmetry and not odd_bool):
                    child_context = parent_context + nuc
                    
                context_list[array_index] = child_context
                
                # Init theta
                init_theta = np.array([0, 0, 0])
                init_p_vec = 0
                init_indicator = np.array([0, 0, 0])
                
    
                parent_p_vec = set_p_vec_sample_array[array_index, first_sample, :]
                success = False
                while not success:
                    if not zero_init:
                        init_theta = np.random.uniform(low = -0.7, high = 0.7, size = 3)
                    init_p_vec = parent_p_vec * np.exp(init_theta)
                    # make sure this p_vec is valid
                    if np.sum(init_p_vec) < 1 and min(init_p_vec) > 0:
                        success = True 
                
                theta_array[array_index] = init_theta
                p_vec_array[array_index] = init_p_vec
                
                # Init indicator
                if not zero_init:
                    init_indicator = np.random.randint(2, size = 3)
                    indicator_array[array_index] = init_indicator

                # Init sigma array, theta probabilities, alpha probabilities
                count = 0
                for j in init_indicator:
                    sigma = slab_sigma
                    alpha_prob = slab_alpha_prob
                    if j == 0:
                        sigma = spike_sigma
                        alpha_prob = spike_alpha_prob
                    sub_theta_probability = -np.log(sigma) - ((init_theta[count]/sigma)**2)/2.0
                    
                    theta_probabilities_array[array_index][count] = sub_theta_probability
                    alpha_probabilities_array[array_index][count] = alpha_prob
                    count += 1
                # get leaf_counts
                leaf_contexts = get_leaf_contexts(child_context, max_mer, oppo_asymmetry)
                leaf_counts = []
                if C_bool:
                    leaf_counts = get_adjusted_leaf_counts(leaf_contexts, leaf_count_dict).astype(np.int32)
                else:
                    leaf_counts = np.array([np.array(leaf_count_dict[x][0:4]) for x in leaf_contexts]).flatten().astype(np.int32)
                
                leaf_count_array[array_index] = leaf_counts

                index_mod += 1

    return theta_array, p_vec_array, indicator_array, context_list, leaf_count_array, theta_probabilities_array, alpha_probabilities_array


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


def find_siblings(node_index):
    
    node_age = node_index - ((node_index - 1) % 4) -1 

    all_siblings = [node_age + sibling_mod for sibling_mod in range(1,5)]
    all_siblings.remove(node_index)

    return all_siblings
 
