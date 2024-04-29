#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions that implement the objective function and a 
discrete optimizer based on local search

Created on Dec 15, 2023

@author: juanjosealcaraz

"""

import system.parameters as par
import model.random_access_model as ra
import model.traffic_model as tm
from system.carrier import compute_offsets
import pickle
import numpy as np

# initialization values
msg3_RUs = [1, 2, 4]
N_sfs_list = [4, 12, 48]

# probabilities of detecting k preambles given n contenders and nsc preambles in the NPRACH
with open('p_detections.pickle', 'rb') as file:
    p_detections = pickle.load(file)

rate_dictionaries = [[], [], []]
resource_dictionaries = [{}, {}, {}]

w_i = par.control_default_values['rar_window']
RAR_window_size = par.RAR_WindowSize_list[w_i]   


def max_msg3_rate(window_RUs, RUs, window_DCIs, DCIs):
    carrier_limit = np.floor(max(0,window_RUs)/RUs)
    DCI_limit = np.floor(max(0, window_DCIs)/DCIs)
    max_msg3 = min(carrier_limit, DCI_limit)
    return max_msg3


def RAR_resources_per_window(period, N_sfs, RUs, ce):
    '''
    this function estimates the maximum number of msg3 messages that 
    can be sent for each CE level in the msg3 window
        
    RAR_window_sf # sfs per RAR window
    DCIs  # max DCIs per window
    max_m # max msg3s sent per per window

    '''  
    max_sfs = period - N_sfs - 4
    max_periods = np.floor(max_sfs/par.NPDCCH_period)
    RAR_p = min(max_periods, RAR_window_size)
    DCIs = par.NPDCCH_sf * RAR_p
    RAR_window_sf = RAR_p * par.NPDCCH_period
    # RAR_window_sf = max(1, RAR_p - 1) * par.NPDCCH_period
    
    max_m = max_msg3_rate(RAR_window_sf, RUs, DCIs, par.NPDCCH_sf_per_CE[ce])

    return max_m, DCIs, RAR_window_sf


def compute_largest_rate(nsc_i, period_i, k_max, debug = False):
    '''
    This function estimates the maximum RA detection rate per CE level
    '''
    max_rate = 0
    nsc_i = 4 * nsc_i # transform into NPRACH subcarriers
    max_detections = 0
    for n in range(1, nsc_i + 1):
        average_detections_for_n = 0
        for k in range(1, n+1):
            average_detections_for_n += min(k,k_max) * p_detections[(k, n, nsc_i)]
        if average_detections_for_n > max_detections:
            max_detections = average_detections_for_n
    max_rate = max_detections / period_i

    if debug:
        print(f'max_rates: {max_rate}')
    
    return max_rate


def nprach_resources(nsc_i, period_i, sfs_i):
    '''
    Computes the NPRACH usage level
    '''
    NPRACH_occupation_rate = (nsc_i * sfs_i) /(12 * period_i)  
    return NPRACH_occupation_rate


def ra_resources(nsc_i, period_i, sfs_i, CE_rate_i, msg3_sf_i):
    '''
    Computes the subframe occupation rate for RA transmissions in a CE level
    in terms of NPRACH resources and RUs for msg3 transmissions
    CE_rate_i provides the user access rate for the CE level in users per subframe
    '''
    NPRACH_occupation_rate = (nsc_i * sfs_i) /(12 * period_i)
    msg3_occupation_rate = CE_rate_i * msg3_sf_i
    return NPRACH_occupation_rate + msg3_occupation_rate


def create_dictionaries(ce, RUs, N_sfs):
    rate_per_conf = {}
    resources_per_conf = {}
    for p_i, period in enumerate(par.period_list):
        for n_i, nsc in enumerate(par.N_sc_list):
            k_max, _, _ = RAR_resources_per_window(period, N_sfs, RUs, ce)
            rate = compute_largest_rate(nsc, period, k_max)
            rate_per_conf[(n_i, p_i)] = rate
            resources_per_conf[(n_i, p_i)] = ra_resources(nsc, period, N_sfs, rate, RUs)
            # resources_per_conf[(n_i, p_i)] = nprach_resources(nsc, period, N_sfs)
    return rate_per_conf, resources_per_conf


# Function to create an ordered dictionary (sorted by values)
def create_ordered_dict(my_dict):
    # Sort the dictionary by its values and return a list of tuples.
    return sorted(my_dict.items(), key=lambda x: x[1])


def remove_items(dict_A, dict_B):
    keys_to_remove = set()

    for key1 in dict_A:
        for key2 in dict_A:
            if dict_A[key2] > dict_A[key1] and dict_B[key2] <= dict_B[key1]:
                keys_to_remove.add(key1)
                break

    for key in keys_to_remove:
        del dict_A[key]

    return dict_A


def update_parameters(msg3_RUs, N_sfs_list, msg3_detection):
    msg3_RUs = msg3_RUs
    N_sfs_list = N_sfs_list
    msg3_detection = msg3_detection
    update_dictionaries(N_sfs_list, msg3_RUs)


def update_dictionaries(N_sfs_list, msg3_RUs):
    '''
    updates the rate and resource dictionaries
    should be used for initialization and whenever N_sfs_lists and/or msg3_RUs change
    The N_sfs_list contains the number of subframes of each CE NPRACH 
    '''
    for ce in range(3):
        rate_d, res_d = create_dictionaries(ce, msg3_RUs[ce], N_sfs_list[ce])
        rate_d = remove_items(rate_d, res_d)
        rate_dictionaries[ce] = create_ordered_dict(rate_d)
        resource_dictionaries[ce] = res_d


def find_conf(ordered_dict, rate):
    # Iterate through sorted key-value pairs.
    for conf, value in ordered_dict:
        # Check if the value is greater than N.
        if value > rate:
            # Return the key as soon as a value greater than N is found.
            return conf
    (conf, rate) = ordered_dict[-1]
    # Return the largest one if no suitable value is found.
    return conf


def configurator(msg3_detection, arrival_rate_per_CE, rate_dictionaries = rate_dictionaries):
    '''
    inputs:
    msg3_detection # detection rate
    arrival_rate_per_CE 

    output:
    (nsc0, nsc1, nsc2, per0, per1, per2)
    '''
    conf = [0, 0, 0, 0, 0, 0]
    for ce in range(3):
        m3_r = msg3_detection[ce]
        a_r = arrival_rate_per_CE[ce]
        target_r = a_r / m3_r
        ordered_rates = rate_dictionaries[ce]
        (nsc, period) = find_conf(ordered_rates, target_r)
        conf[ce] = nsc
        conf[ce + 3] = period
    return conf


def nprach_configurator(th_C1, th_C0, N_sfs_list, arrival_rate_per_CE, probabilities):
    msg3_RUs = ra.compute_msg3_rus(th_C1, th_C0)
    msg3_detection = ra.msg3_detection_estimation(th_C1, th_C0, probabilities)          
    resources = 0.0
    slack = 0.0
    conf = [0, 0, 0, 0, 0, 0]
    CAPABLE = True
    for ce in range(3):
        # if not arrival_rate_per_CE[ce]:
        #     continue              
        # dictionaries
        rate_d, res_d = create_dictionaries(ce, msg3_RUs[ce], N_sfs_list[ce])
        rate_d = remove_items(rate_d, res_d)
        o_rate_d = create_ordered_dict(rate_d)
        
        # find right conf
        m3_r = msg3_detection[ce]
        a_r = arrival_rate_per_CE[ce]
        target_r = a_r / m3_r if m3_r else a_r
        (nsc, period) = find_conf(o_rate_d, target_r)
        conf[ce] = nsc
        conf[ce + 3] = period

        # resources and rate
        resources_i = res_d[(nsc, period)]
        rate_i = rate_d[(nsc, period)]
        if rate_i < target_r:
            CAPABLE = False
        slack += (target_r - rate_i)**2
        resources += resources_i
        
    return conf, resources, slack, CAPABLE

def full_configurator(arrival_rate, probabilities):
    '''
    inputs:
    th_pairs: all the CE threshold pairs (th_C1, th_C0) that should be considered
    arrival rate: full arrival rate (users per subframe)
    params: estimated distribution parameters of the users
    period_list: NPRACH period values
    N_sc_list: NPRACH subcarrier values

    output:
    (th_C1, th_C0) (nsc0, nsc1, nsc2, per0, per1, per2)
    '''
    config_dict = {}

    for (th_C1, th_C0) in par.th_pairs:
        N_sfs_list = ra.compute_NPRACH_sf(th_C1, th_C0)
        arrival_rate_per_CE = tm.arrival_flows_estimation(th_C1, th_C0, arrival_rate, probabilities, ra.th_values)
        conf, conf_res, conf_slack, CAPABLE = nprach_configurator(th_C1, th_C0, N_sfs_list, arrival_rate_per_CE, probabilities)
        
        # compare
        nsc = [par.N_sc_list[i] for i in conf[:3]]
        periods = [par.period_list[i] for i in conf[3:]]
        _, _, _, valid = compute_offsets(periods, N_sfs_list, nsc)

        if valid:
            config_dict[(th_C1, th_C0)] = (conf_res, conf_slack, (th_C1, th_C0), conf, CAPABLE)   

    # Step 1: Find the full configuration with less occupation that is CAPABLE
    filtered_items = [(res, slack, th, conf) for (res, slack, th, conf, c_) in config_dict.values() if c_]
    if filtered_items:
        result = min(filtered_items, key=lambda item: item[0])
    else:
        # Step 2: If none is CAPABLE, find the configuration with the minimum distance to the target
        result = min(config_dict.values(), key=lambda item: item[1])       
    
    th =result[2]
    conf = result[3]
    
    return th, conf
