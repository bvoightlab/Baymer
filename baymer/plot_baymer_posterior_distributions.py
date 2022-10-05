#!/usr/bin/Python

# Created by: Christopher J Adams 9/7/2022
# 

###############################################################################
###
### This script will plot baymer posterior distributions and summarize the data
###
###############################################################################


## NOTE Assumes symmetric-ish counts

#import cProfile
import sys
import getopt
import os
import yaml
import json
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import collections as mc
matplotlib.rcParams['agg.path.chunksize'] = 10000
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def help(exit_num=1):
    print("""-----------------------------------------------------------------
ARGUMENTS
    -c => <yaml> config file REQUIRED
    -t => <boolean> plot theta plots OPTIONAL
""")
    sys.exit(exit_num)

###############################################################################
###########################  COLLECT AND CHECK ARGUMENTS  #####################
###############################################################################

## GLOBAL VARS

## MAIN ##

def main(argv): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:], "c:t")
    except getopt.GetoptError:
        print("Error: Incorrect usage of getopts flags!")
        help()

    options_dict = dict(opts)

    ## Required arguments
    try:
        config_file = options_dict['-c']

    except KeyError:
        print("Error: One of your required arguments does not exist.")
        help()
    
    plot_thetas = options_dict.get('-t', False)
    if plot_thetas == '':
        plot_thetas = True
    

    print("Acceptable Inputs Given")
    
    

    driver(config_file, skip_thetas)


###############################################################################
#############################  DRIVER  ########################################
###############################################################################


## drive the script ##
## ONE-TIME CALL -- called by main

def driver(config_file, plot_thetas = False):

    config_dict = yaml.load(open(config_file, 'r'), Loader=yaml.SafeLoader)
    
    #train_data_list = ['EVEN', 'ODD']
    # gather overall info from config dict
    max_layer = config_dict['max_mer'] - 1
    pop = config_dict['pop']
    feature = config_dict['feature']
    posterior_dir = config_dict['posterior_dir']
    random_seeds = config_dict['random_seeds']
    dataset = config_dict['dataset']
    c = config_dict['c']
    summary_posterior_dict = {dataset: {}}
    for layer in range(0, max_layer + 1):
        print("Layer ", layer)
        summary_posterior_dict[dataset][layer] = {}
        # gather layer data
        index_dict = posterior_dir + "index_dict.layer_" + str(layer) + ".json"

        index_context_dict = open_json_dict(index_dict)
        num_iterations = config_dict[layer]["num_iterations"]
        
        burnin = config_dict[layer]["burnin"]
        thinning_parameter = config_dict[layer]["thinning_parameter"]
        thinned_burnin = 0

        # make output directory
        dir_name = "layer_" + str(layer)
        layer_output_dir = output_dir + dir_name
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass

        try:
            os.mkdir(layer_output_dir)
        except FileExistsError:
            pass

        likelihood_data_list = []
        p_data_list = []  
        theta_data_list = []
        ind_data_list = []
        # first thetas and p_vec
        if plot_thetas:
            for random_seed in random_seeds:
                theta_chain_matrix_file = "{}{}_{}_{}_rs{}_thetas.burned_in.thinned.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
                theta_data_list.append(theta_chain_matrix_file)
        
                p_vec_file = "{}{}_{}_{}_rs{}_rate_matrix.burned_in.thinned.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
                p_data_list.append(p_vec_file)

                ind_file = "{}{}_{}_{}_rs{}_indicator.burned_in.thinned.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
                ind_data_list.append(ind_file)
            theta_out_dir = "{}/theta_out_plots/".format(layer_output_dir)
            try:
                os.mkdir(theta_out_dir)
            except FileExistsError:
                pass

            plot_theta_chain(layer, random_seeds, theta_data_list, p_data_list, ind_data_list, theta_out_dir, index_context_dict, true_values_dict, summary_posterior_dict[dataset][layer], thinned_burnin, plot_thetas)
            print('thetas plotted')
        # next sigmas, alphas, and indicators
        if layer != 0:
            skip_sigma = False
            sigma_data_list = []
            for random_seed in random_seeds:
                sigma_chain_matrix_file = "{}{}_{}_{}_rs{}_sigma_matrix.burned_in.thinned.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
                if os.path.isfile(sigma_chain_matrix_file):
                    sigma_data_list.append(sigma_chain_matrix_file)
                else:
                    skip_sigma = True
            if not skip_sigma:
                plot_sigma_chain(random_seeds, sigma_data_list, layer_output_dir, c, summary_posterior_dict[dataset][layer], thinned_burnin)
            
            alpha_data_list = []
            ind_data_list = []
            for random_seed in random_seeds:
                alpha_chain_matrix_file = "{}{}_{}_{}_rs{}_alphas.burned_in.thinned.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
                alpha_data_list.append(alpha_chain_matrix_file)
                indicator_chain_matrix_file = "{}{}_{}_{}_rs{}_indicator.burned_in.thinned.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
                ind_data_list.append(indicator_chain_matrix_file)
              
            plot_alpha_chain(random_seeds, alpha_data_list, ind_data_list, layer_output_dir, summary_posterior_dict[dataset][layer], thinned_burnin)
            
            plot_indicator_distribution(random_seeds, ind_data_list, layer_output_dir, index_context_dict, summary_posterior_dict[dataset][layer], thinned_burnin)
        # finally likelihoods
        for random_seed in random_seeds:
            likelihood_chain_matrix_file = "{}{}_{}_{}_rs{}_posterior_matrix.full_trace.layer_{}.npy".format(posterior_dir, pop, feature, dataset, random_seed, layer)
            likelihood_data_list.append(likelihood_chain_matrix_file)
        
        plot_likelihoods(random_seeds, likelihood_data_list, burnin, thinning_parameter, layer_output_dir, summary_posterior_dict[dataset][layer])
        
    posterior_dict_out_file = "{}/{}.{}.{}.final_posterior_summaries.json".format(output_dir, pop, feature, dataset)
    with open(posterior_dict_out_file, 'w') as jFile:
        json.dump(summary_posterior_dict, jFile)
    

def plot_likelihoods(chain_lists, data_list, burnin, thinning_parameter, output_dir, summary_posterior_dict):

    likelihood_label_dict = {0: 'theta probability',
                             1: 'alpha probability',
                             2: 'leaf likelihood',
                             3: 'set_probability',
                             4: 'posterior'}

    all_chains = []
    summary_posterior_dict['likelihood'] = []
    y_vals_dict = {}
    row_count = 0
    total_chains = len(chain_lists)
    for i in range(total_chains):
        
        chain = chain_lists[i]
        y_vals_dict[chain] = {}
        chain_label = "rs{}".format(chain)
        mat = data_list[i]
        data_matrix = np.load(mat)
        thinned_burned_in_likelihoods = data_matrix[burnin::thinning_parameter]
        if i == 0:
            all_chains = thinned_burned_in_likelihoods
        else:
            all_chains = np.append(all_chains, thinned_burned_in_likelihoods)
            if i == total_chains - 1:
                summary_posterior_dict['likelihood'] = float(np.mean(all_chains))
        
        transposed_data = data_matrix.T
        x_vals = list(range(len(transposed_data[0])))
        row_count = 0
        for row in transposed_data:
            likelihood_type = likelihood_label_dict[row_count]
            y_vals_dict[chain][likelihood_type] = list(row)
            row_count += 1

    fig_width = 15
    fig_height = row_count * 10
    fig = plt.figure(figsize = (fig_width, fig_height))
    grid = plt.GridSpec(row_count, 2, hspace=0.2, wspace=0.2)
    #print(x_vals)
    row_count = 0
    for row in list(likelihood_label_dict.keys()):
        likelihood_type = likelihood_label_dict[row]
        ax1 = fig.add_subplot(grid[row,0])
        ax2 = fig.add_subplot(grid[row,1])
        if row_count == 4:
            ax1.set(xlabel = "Iteration")
            ax2.set(xlabel = "Iteration")
        ax1.set(ylabel = "Probability")
            
        for chain in chain_lists:
            chain_label = "rs{}".format(chain)
            y_vals = y_vals_dict[chain][likelihood_type]
            #print(y_vals)
            ax1.plot(x_vals, y_vals, label = chain)
            ax1.set_title(likelihood_type)
            ax2.plot(x_vals[burnin:], y_vals[burnin:], label = chain_label)
            burned_in_title = "{} {} burned in iterations".format(likelihood_type, burnin)
            ax2.set_title(burned_in_title)
        ax1.legend()
        ax2.legend()
    output_file = "{}/posterior_likelihoods.png".format(output_dir)
   
    plt.savefig(output_file)
 

def plot_theta_chain(layer, chain_lists, theta_data_list, p_data_list, ind_data_list, output_dir, index_context_dict, true_values_dict, summary_posterior_dict, thinned_burnin, plot_thetas):
    
    
    fig, axs = plt.subplots(4, figsize = (10, 30))
    
    mut_nuc_index_look_up = ['A', 'C', 'G', 'T']
    mut_indices = []
    num_chains = len(chain_lists)
    master_data_dict = {}
    mut_ground_truth_dict = {}
    x_vals = None
    posterior_distribution_dict = {'theta': {},
                                   'p': {},
                                   'ind': {}}
    
    empirical_p_context_dict = {}
    p_vec_all_chains_dict = {}
    theta_all_chains_dict = {}
    ind_all_chains_dict = {}
    
    summary_posterior_dict['theta'] = {}
    summary_posterior_dict['p_vec'] = {}
    no_ind = False
    for i in range(num_chains):
        chain = chain_lists[i]
        chain_label = "rs{}".format(chain)
        mat = theta_data_list[i]
        theta_data_matrix = np.load(mat)
        mean_thetas = np.mean(theta_data_matrix[thinned_burnin:, :, :], axis = 0).flatten()
        print(len(mean_thetas))

        x_vals = range(len(theta_data_matrix))
        axs[0].hist(mean_thetas, alpha = 0.5, label = chain_label, bins = np.arange(-2.5, 2.5, 0.01))
        axs[1].hist(theta_data_matrix[thinned_burnin:, :, :].flatten(), alpha = 0.5, label = chain_label, bins = np.arange(-2.5, 2.5, 0.01))
        axs[2].hist(theta_data_matrix[0, :, :].flatten(), alpha = 0.5, label = chain_label, bins = np.arange(-2.5, 2.5, 0.01))
        axs[3].hist(theta_data_matrix[-1, :, :].flatten(), alpha = 0.5, label = chain_label, bins = np.arange(-2.5, 2.5, 0.01))
        mat = p_data_list[i]
        
        p_data_matrix = np.load(mat)
        
        try:
            mat = ind_data_list[i]
            ind_data_matrix = np.load(mat)
        except FileNotFoundError:
            no_ind = True
            
        print(no_ind)    

        index = 0
        open_dict = False
        mer_level = 0
        previous_context_length = -1
        #print('++++++++')
        true_diff_list = []
        emp_diff_list = []
        #for row in transposed_data:
        for index in range(len(theta_data_matrix[0])):
            #theta_row = [list(x[index]) for x in theta_data_matrix]
            #p_row = [list(x[index]) for x in p_data_matrix]
            theta_row = theta_data_matrix[thinned_burnin:, index, :]
            p_row = p_data_matrix[thinned_burnin:, index, :]            
            ind_row = []
            if not no_ind:
                ind_row = ind_data_matrix[thinned_burnin:, index, :]
            context_string = index_context_dict[str(index)]
            
            if i == 0:
                p_vec_all_chains_dict[context_string] = p_row
                theta_all_chains_dict[context_string] = theta_row
                if not no_ind:
                    ind_all_chains_dict[context_string] = ind_row
            else:
                p_vec_all_chains_dict[context_string] = np.append(p_vec_all_chains_dict[context_string], p_row, axis = 0)
                theta_all_chains_dict[context_string] = np.append(theta_all_chains_dict[context_string], theta_row, axis = 0)
                if not no_ind:
                    ind_all_chains_dict[context_string] = np.append(ind_all_chains_dict[context_string], ind_row, axis = 0)
            if i == num_chains - 1:
                summary_posterior_dict['theta'][context_string] = {}
                summary_posterior_dict['p_vec'][context_string] = {}
                theta_array = theta_all_chains_dict[context_string] 
                p_vec_array = p_vec_all_chains_dict[context_string] 

                summary_posterior_dict['theta'][context_string]['mean'] = np.mean(theta_array, axis = 0).tolist()
                summary_posterior_dict['p_vec'][context_string]['mean'] = np.mean(p_vec_array, axis = 0).tolist()

                
            


            if not plot_thetas:
                continue
            center_nuc = context_string[int(len(context_string)/2)]
            if len(context_string) % 2 == 0:
                center_nuc = context_string[int(len(context_string)/2) - 1]
            mut_indices = [0, 2, 3]
            if center_nuc == 'A':
                mut_indices = [1, 2, 3]
            #theta_posterior_distribution = theta_row
            #p_posterior_distribution = thinned_p_row
            context_length = len(context_string)
            if context_length > previous_context_length:
                open_dict = False
                previous_context_length = context_length
            if not open_dict:
                count_ground_truth_dict_file = true_values_dict[str(context_length) + 'mer']['context_counts']
                count_ground_truth_dict = open_json_dict(count_ground_truth_dict_file)

            theta_index = 0
            for context_base in mut_indices:
                mut_string = context_string + ">" + mut_nuc_index_look_up[context_base]
                try:
                    sub_theta_posterior = theta_row[:, theta_index]
                    #sub_theta_posterior = [x[theta_index] for x in theta_posterior_distribution]
                    sub_p_posterior = p_row[:, theta_index]
                    #sub_p_posterior = [x[theta_index] for x in p_posterior_distribution]
                    sub_ind_posterior = []
                    if not no_ind:
                        sub_ind_posterior = ind_row[:, theta_index]
                except TypeError:
                    print("Error!!!!: ", mut_string)
                    continue 
                
                empirical_p = count_ground_truth_dict[context_string][context_base] / count_ground_truth_dict[context_string][4]
                empirical_p_context_dict[mut_string] = empirical_p
                 
                ## add to data objects
                try:
                    posterior_distribution_dict['theta'][context_string][mut_string][i] = sub_theta_posterior
                    posterior_distribution_dict['p'][context_string][mut_string][i] = sub_p_posterior
                    if not no_ind:
                        posterior_distribution_dict['ind'][context_string][mut_string][i] = sub_ind_posterior
                except KeyError:
                    try:
                        posterior_distribution_dict['theta'][context_string][mut_string] = [0] * num_chains
                        posterior_distribution_dict['theta'][context_string][mut_string][0] = sub_theta_posterior
                        posterior_distribution_dict['p'][context_string][mut_string] = [0] * num_chains
                        posterior_distribution_dict['p'][context_string][mut_string][0] = sub_p_posterior
                        if not no_ind:
                            posterior_distribution_dict['ind'][context_string][mut_string] = [0] * num_chains
                            posterior_distribution_dict['ind'][context_string][mut_string][0] = sub_ind_posterior
                    except KeyError:
                        posterior_distribution_dict['theta'][context_string] = {}
                        posterior_distribution_dict['theta'][context_string][mut_string] = [0] * num_chains
                        posterior_distribution_dict['theta'][context_string][mut_string][0] = sub_theta_posterior
                        posterior_distribution_dict['p'][context_string] = {}
                        posterior_distribution_dict['p'][context_string][mut_string] = [0] * num_chains
                        posterior_distribution_dict['p'][context_string][mut_string][0] = sub_p_posterior
                        if not no_ind:
                            posterior_distribution_dict['ind'][context_string] = {}
                            posterior_distribution_dict['ind'][context_string][mut_string] = [0] * num_chains
                            posterior_distribution_dict['ind'][context_string][mut_string][0] = sub_ind_posterior

                theta_index += 1
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    file_name = "{}/theta_distribution_histograms.layer_{}.png".format(output_dir, layer)
    plt.savefig(file_name)
    plt.close(fig)
    
    if not plot_thetas:
        return
    print('+++++++')
    # all data has been gathered
    for context_string in posterior_distribution_dict['theta']:
       
        fig, axs = plt.subplots(3,3, figsize = (25, 25))
        if not no_ind:
            plt.close(fig)
            fig, axs = plt.subplots(4,3, figsize = (25, 35))
        plt.subplots_adjust(top=0.85)
        mean_chain = 0
        
        mut_string_index = 0
        for mut_string in posterior_distribution_dict['theta'][context_string]:
            
            axs[2, mut_string_index].set(xlabel = "Iteration")
            if not no_ind:
                axs[3, mut_string_index].set(xlabel = "Iteration")
            if mut_string_index == 0:
                axs[0, 0].set(ylabel = "Count")
                axs[1, 0].set(ylabel = "Count")
                axs[2, 0].set(ylabel = "Theta")
                if not no_ind:
                    axs[3,0].set(ylabel = "Indicator average")
            for chain_index in range(len(chain_lists)):
                theta_y_vals = posterior_distribution_dict['theta'][context_string][mut_string][chain_index]
                p_y_vals = posterior_distribution_dict['p'][context_string][mut_string][chain_index]
                chain_label = "rs_{}".format(chain_lists[chain_index])
                
                axs[0, mut_string_index].hist(p_y_vals, alpha = 0.5, label = chain_label)
                axs[0, mut_string_index].set(xlabel = "multinomial p")
                axs[0, mut_string_index].set_title(mut_string)
                axs[1, mut_string_index].hist(theta_y_vals, alpha = 0.5, label = chain_label)
                axs[1, mut_string_index].set(xlabel = "Theta")
 
                axs[2, mut_string_index].plot(x_vals, theta_y_vals, label = chain_label)
                if not no_ind:
                    ind_y_vals = posterior_distribution_dict['ind'][context_string][mut_string][chain_index]
                    pad_ind_y_vals = np.append(np.zeros(250), ind_y_vals, axis = 0)
                    pad_ind_y_vals = np.append(pad_ind_y_vals, np.zeros(249), axis = 0)
                    moving_average_y_vals = moving_average(pad_ind_y_vals, n=500)
                    axs[3,mut_string_index].plot(x_vals, moving_average_y_vals, label = chain_label)
                   
                chain_index += 1
                if mut_string_index == 0:
                    axs[0, 0].legend()
                    axs[1, 0].legend()
                    axs[2, 0].legend()
                    if not no_ind:
                        axs[3,0].legend()

            emp_p = empirical_p_context_dict[mut_string]
            axs[0, mut_string_index].axvline(x = emp_p, linestyle = 'dashed', label = "empirical estimate".format(chain_label))

            mut_string_index += 1
        
        file_name = "{}/theta_{}_posterior_plots.png".format(output_dir, context_string)
        plt.savefig(file_name)
        
        plt.close(fig)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_alpha_chain(chain_lists, alpha_data_list, ind_data_list, output_dir, summary_posterior_dict, thinned_burnin):
    
    num_chains = len(chain_lists)
    
    fig, axs = plt.subplots(3, figsize = (12, 16))
    plt.subplots_adjust(top=0.85)
    
    axs[0].set(xlabel = "alpha")
    axs[0].set(ylabel = "count")
    axs[1].set(ylabel = "alpha")
    axs[1].set(xlabel = "Iteration")
    axs[2].set(ylabel = "Indicator ratio")
    axs[2].set(xlabel = "Iteration")
 
    all_alpha_chains = []
    for i in range(num_chains):
        chain = chain_lists[i]
        chain_label = "rs{}".format(chain)
        mat = alpha_data_list[i]
        alpha_data_array = np.load(mat)[thinned_burnin:]
        ind_mat = ind_data_list[i]
        ind_data_array = np.load(ind_mat)[thinned_burnin:]
        total_sub_thetas = len(ind_data_array[0]) * 3
        
        index = 0
        mer_level = 0
        if i == 0:
            all_alpha_chains = alpha_data_array
        else:
            all_alpha_chains = np.append(all_alpha_chains, alpha_data_array)
        if i == num_chains - 1:
            summary_posterior_dict['alpha'] = float(np.mean(all_alpha_chains))
        axs[0].hist(alpha_data_array, alpha = 0.5, label = chain_label, bins = np.arange(0, 1.001, 0.001))
        
        x_vals = range(len(alpha_data_array))
        axs[1].plot(x_vals, alpha_data_array, label = chain_label)
        mean_chain = float(np.mean(alpha_data_array))
        axs[1].axhline(y=mean_chain, linestyle = '--', label = "mean {}".format(chain_label))

        ind_ratios = [np.sum(x) / total_sub_thetas for x in ind_data_array]
        axs[2].plot(x_vals, ind_ratios, label = chain_label)
    max_alpha = max(all_alpha_chains)
    min_alpha = min(all_alpha_chains)

    axs[0].set_xlim(max([0, min_alpha - 0.05]), min([1.0, max_alpha + 0.05]))
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()


    file_name = "{}/alpha_posterior_plots.png".format(output_dir)
    plt.savefig(file_name)
    plt.close(fig)

def plot_sigma_chain(chain_lists, sigma_data_list, output_dir, c, summary_posterior_dict, thinned_burnin):
    
    num_chains = len(chain_lists)
    
    fig, axs = plt.subplots(3, figsize = (12, 20))

    plt.subplots_adjust(top=0.85)
    
    axs[0].set(xlabel = "slab sigma")
    axs[0].set(ylabel = "Density")
    ax_0_x_vals = np.linspace(-2,2, 100000)
    
    axs[1].set(xlabel = "slab sigma")
    axs[1].set(ylabel = "count")
    all_chains = []
    for i in range(num_chains):
        chain = chain_lists[i]
        chain_label = "rs{}".format(chain)
        mat = sigma_data_list[i]
        sigma_data_array = np.load(mat)[thinned_burnin:]
        if i == 0:
            all_chains = sigma_data_array
        else:
            all_chains = np.append(all_chains, sigma_data_array)
        if i == num_chains - 1:
            summary_posterior_dict['slab_sigma'] = float(np.mean(all_chains))
        # plot the first ax
        mean_slab_sigma = np.mean(sigma_data_array)
        ax_0_slab_y_vals = norm.pdf(ax_0_x_vals, 0, mean_slab_sigma)
        ax_0_spike_y_vals = norm.pdf(ax_0_x_vals, 0, c * mean_slab_sigma)
        
        axs[0].plot(ax_0_x_vals, ax_0_slab_y_vals, label = 'slab_{}'.format(chain_label))
        axs[0].plot(ax_0_x_vals, ax_0_spike_y_vals, label = 'spike_{}'.format(chain_label))
        ylim = max(ax_0_slab_y_vals) * 2
        axs[0].set_ylim([0, ylim])
        axs[1].hist(sigma_data_array, alpha = 0.5, label = chain_label, bins = np.arange(0, 5.001, 0.001))
        
        x_vals = range(len(sigma_data_array))
        axs[2].plot(x_vals, sigma_data_array, label = chain_label)
        mean_chain = np.mean(sigma_data_array)
        axs[2].axhline(y=mean_chain, linestyle = '--', label = "mean {}".format(chain_label))
    
    max_sigma = max(all_chains)
    min_sigma = min(all_chains)
    axs[1].set_xlim(max([0, min_sigma - 0.01]), max_sigma + 0.01)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()


    file_name = "{}/sigma_posterior_plots.png".format(output_dir)
    plt.savefig(file_name)
    plt.close(fig)
                 
def plot_indicator_distribution(chain_lists, ind_data_list, output_dir, index_context_dict, summary_posterior_dict, thinned_burnin):

    num_chains = len(chain_lists)
    all_chains_indicator_dict = {}
    summary_posterior_dict['ind'] = {}
    fig, axs = plt.subplots(3, figsize = (10, 20))
    
    plt.subplots_adjust(top=0.85)
    
    ax_per_slab = axs[0]
    ax_per_slab.set(xlabel = "% slab")
    ax_per_slab.set(ylabel = "Count")
    
    ax_per_flip = axs[1]
    ax_per_flip.set(xlabel = "% iterations indicator flipped")
    ax_per_flip.set(ylabel = "Count")

    ax_variable = axs[2]
    ax_variable.set(ylabel = "% thetas switched")
    ax_variable.set(xlabel = 'Iteration')

    for i in range(num_chains):
        chain = chain_lists[i]
        chain_label = "rs{}".format(chain)
        mat = ind_data_list[i]
        ind_data_array = np.load(mat)
        total_contexts = len(ind_data_array[0])
        total_thetas = total_contexts * 3
        total_iterations = len(ind_data_array)
        hist_vals = np.zeros(total_thetas)
        total_flips = np.zeros(total_thetas)
        change_per_iteration = np.zeros(total_iterations)
        for index in range(total_contexts):
            context_string = index_context_dict[str(index)]
            theta_ind_array = ind_data_array[thinned_burnin:, index, :]
            if i == 0:
                all_chains_indicator_dict[context_string] = theta_ind_array
            else:
                all_chains_indicator_dict[context_string] = np.append(all_chains_indicator_dict[context_string], theta_ind_array, axis = 0)
            if i == num_chains - 1:
                summary_posterior_dict['ind'][context_string] = (np.sum(all_chains_indicator_dict[context_string], axis = 0) / float(total_iterations * num_chains)).tolist()
                
            for j in range(3):
                ind_array = theta_ind_array[:, j]
                percent_slab = float(np.sum(ind_array)) / float(total_iterations)
                hist_vals[index * 3 + j] = percent_slab
                
                current_ind = ind_array[0]
                it = 0
                for k in ind_array:
                    
                    if k != current_ind:
                        total_flips[index * 3 + j] += 1
                        current_ind = k
                        change_per_iteration[it] += 1
                    it += 1

        percent_flips = total_flips / total_iterations

        percent_change = change_per_iteration / total_thetas 
        x_vals = range(total_iterations)
                    
        # plot the first ax
        ax_per_slab.hist(hist_vals, alpha = 0.5, bins = np.arange(0, 1.01, 0.01), label = chain_label)
        
        # plot the second ax
        ax_per_flip.hist(percent_flips, alpha = 0.5, bins = np.arange(0, 1.01, 0.01), label = chain_label)

        # plot the third ax
        ax_variable.plot(x_vals, percent_change, label = chain_label)

    ax_per_slab.legend()
    ax_per_flip.legend()
    ax_variable.legend()

    file_name = "{}/indicator_posterior_plots.png".format(output_dir)
    plt.savefig(file_name)
    plt.close(fig)

def open_json_dict(f):

    with open(f, 'r') as jFile:
        d = json.load(jFile)

    return d

#######################################################################################################################################################
#cProfile.run('main(sys.argv[1:])')
         
if __name__ == "__main__":
    main(sys.argv[1:])
