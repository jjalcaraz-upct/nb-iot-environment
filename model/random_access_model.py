#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions that implement the NPRACH and the RAR detection model

Created on Dec 5, 2023

@author: juanjosealcaraz

"""
import system.parameters as par
import numpy as np
import pickle
from scipy.stats import lognorm
from model.traffic_model import power_distribution_parameters, arrival_flows_estimation


# detection rates
with open('th_detection_rates.pickle', 'rb') as file:
    th_detection_rates = pickle.load(file)

# msg3 link adaptation parameter configuration
th_values = par.th_values
ext_th_values = [par.threshold_list[0]-1] + par.threshold_list[:-1]

# RAR window size
w_i = par.control_default_values['rar_window']
RAR_window_size = par.RAR_WindowSize_list[w_i]

# number of subrames of each NPRACH
NPRACH_N_sf_list = [int(np.ceil(N_rep * 5.6)) for N_rep in par.N_rep_preamble_list]

# probabilities of detecting k preambles given n contenders and nsc preambles in the NPRACH
with open('p_detections.pickle', 'rb') as file:
    p_detections = pickle.load(file)


def compute_NPRACH_sf(th_C1, th_C0):
    '''
    Computes the number of subframes of the NPRACH for each CE level 
    based on the RSRP thresholds that determine the CE levels
    '''
    min_th = par.threshold_list[0]
    rep_C2 = par.thr_to_n_rep[min_th]
    sf_c2 = NPRACH_N_sf_list[rep_C2]
    if th_C1 < par.no_rep_th:
        th_C1 = max(min_th, th_C1)
        rep_C1 = par.thr_to_n_rep[th_C1]
    else:
        rep_C1 = 0
    sf_c1 = NPRACH_N_sf_list[rep_C1]
    if th_C0 < par.no_rep_th:
        th_C0 = max(min_th, th_C0)
        rep_C0 = par.thr_to_n_rep[th_C0]
    else:
        rep_C0 = 0
    sf_c0 = NPRACH_N_sf_list[rep_C0]
    if th_C0 == 0 or th_C1 == th_C0: # only two CE levels: CE1 and CE2
        sf_c0 = 0
    return [sf_c0, sf_c1, sf_c2]


def compute_msg3_rus(th_C1, th_C0):
    '''
    Computes the number of resources consumed by the msg3 messages sent by
    the devices of each CE level during RAR windows
    '''
    msg3_default_conf = [0, 0, 0]
    rus = [0, 0, 0]
    min_th = par.threshold_list[0]
    msg3_default_conf[2] = par.msg_3_conf[min_th]
    if th_C1 < par.no_rep_msg3_th:
        msg3_default_conf[1] = par.msg_3_conf[th_C1]
    else:
        msg3_default_conf[1] = (2,1,1)

    if th_C0 < par.no_rep_msg3_th:
        msg3_default_conf[0] = par.msg_3_conf[th_C0]
    else:
        msg3_default_conf[0] = (2,1,1)
    
    for i, msg3_conf in enumerate(msg3_default_conf):
        rus[i] = msg3_conf[1]*msg3_conf[2]
    
    return rus


def msg3_detection_estimation(th_1, th_0, probabilities, default = [0.92, 0.9, 0.8]):
    msg3_detection = [0, 0, 0]
    marginal_prob = [0, 0, 0]
    marginal_p = probabilities[-1]
    i = 0
    for i, th in enumerate(th_values):
        if th <= th_1: # CE2
            # print(f' {th} th <= th_1')
            ce = 2
            conf = par.msg_3_conf[th_values[0]]
        elif th > th_1 and th <= th_0: #CE1
            # print(f' {th} th > th_1 and th <= th_0')
            ce = 1
            if th_1 >= par.no_rep_msg3_th:
                conf = (2,1,1)
            else:
                conf = par.msg_3_conf[th_1]
        if th > th_0: #CE0
            # print(f' {th} th >= th_0')
            ce = 0
            if th_0 >= par.no_rep_msg3_th:
                conf = (2,1,1)
            else:
                conf = par.msg_3_conf[th_0]
        th_e = ext_th_values[i]
        detection = th_detection_rates[th_e][conf]
        prob = probabilities[i]
        msg3_detection[ce] += prob * detection
        marginal_prob[ce] += prob
    
    msg3_detection[0] +=  marginal_p * th_detection_rates[th][conf]  
    marginal_prob[0] += marginal_p

    if th_1 != th_0:
        for ce in range(3):
            msg3_detection[ce] = msg3_detection[ce] / marginal_prob[ce] if marginal_prob[ce] > 0 else default[ce]
    else:
        msg3_detection[2] = msg3_detection[2] / marginal_prob[2] if marginal_prob[2] > 0 else default[2]
        msg3_detection[1] = msg3_detection[0] / marginal_prob[0] if marginal_prob[0] > 0 else default[0]
        msg3_detection[0] = 0

    return msg3_detection

# MSG3 RAR WINDOW EFFICIENCY

w_i = par.control_default_values['rar_window']
RAR_window_size = par.RAR_WindowSize_list[w_i]

def max_msg3_rate(window_RUs, RUs, window_DCIs, DCIs):
    carrier_limit = max(0,window_RUs)/RUs
    DCI_limit = max(0, window_DCIs)/DCIs
    max_msg3 = min(carrier_limit, DCI_limit)
    return max_msg3


def RAR_resources_per_window(periods, RUs, N_sfs):
    '''
    this function tries to maximum number of msg3 messages that 
    can be sent for each CE level in the msg3 window

    '''
    RAR_window_sf = [0, 0, 0] # sfs per RAR window
    DCIs = [0, 0, 0] # max DCIs per window
    max_m = [0, 0, 0] # max msg3s sent per per window
    
    for ce in range(3):
        max_sfs = periods[ce] - N_sfs[ce] - 4
        max_periods = np.floor(max_sfs/par.NPDCCH_period)
        RAR_p = min(max_periods, RAR_window_size)
        DCIs[ce] = par.NPDCCH_sf * RAR_p
        RAR_window_sf[ce] = max(1, RAR_p - 1) * par.NPDCCH_period
    
    for ce in range(3):
        max_m[ce] = max_msg3_rate(RAR_window_sf[ce], RUs[ce], DCIs[ce], par.NPDCCH_sf_per_CE[ce])

    return max_m, DCIs, RAR_window_sf


def msg3_rate_estimation(arrival_rates, periods, RUs, N_sfs, debug = False):
    '''
    This function tries to estimate the max msg3 rate in msgs per window for each CE level 
    from the traffic intensity and the available resources per CE level in the RAR window.
    A shortest first scheduling model is assumed
    
    arrivals = detected preambles per NPRACH
    periods = NPRACH periods
    '''
    max_msg3, DCIs, rar_w_sf = RAR_resources_per_window(periods, RUs, N_sfs)

    arrivals = [a_ * p_ for a_, p_ in zip(arrival_rates, periods)]

    # par.NPDCCH_sf_per_CE[ce] # DCIs consumed by an msg3 per CE level

    if debug:
        print(f'max_msg3: {max_msg3}')
        print(f'RUs: {RUs}')
        print(f'rar_w_sf: {rar_w_sf}')
        print(f'arrivals: {arrivals}')

    # print(f'max_msg3: {max_msg3}')

    per_ = [p//80 for p in periods]

    RU_demand = [0,0,0]
    DCI_demand = [0,0,0]
    RU_max_load = [0,0,0]
    DCI_max_load = [0,0,0]
    dci_1 = par.NPDCCH_sf_per_CE[1]
    dci_2 = par.NPDCCH_sf_per_CE[2]
    window = [[], [], []]

    for ce in range(3):
        RU_demand[ce] = arrivals[ce] * RUs[ce]
        DCI_demand[ce] = arrivals[ce] * par.NPDCCH_sf_per_CE[ce]        
        RU_max_load[ce] = max_msg3[ce] * RUs[ce]
        DCI_max_load[ce] = max_msg3[ce] * par.NPDCCH_sf_per_CE[ce]
        for i in range(49):
            if i % per_[ce] == 0:
                window[ce].append(1)
            else:
                window[ce].append(0)

    # CE 0
    RU_consumed = []
    DCI_consumed = []
    msg3_0 = 0
    for win_0 in window[0]:
        RU_consumed_ = win_0 * min(RU_max_load[0], RU_demand[0])
        DCI_consumed_ = win_0 * min(DCI_max_load[0], DCI_demand[0])
        RU_consumed.append(RU_consumed_)
        DCI_consumed.append(DCI_consumed_)
        msg3_0 += win_0 * max_msg3[0]
    msg3_0 = msg3_0/sum(window[0])

    if debug:
        print(RU_consumed)
        print(DCI_consumed)
        print('------------------')

    # CE 1
    msg3_1 = 0
    for i, (RU_0, DCI_0, win_1) in enumerate(zip(RU_consumed, DCI_consumed, window[1])):
        RU_consumed_ = win_1 * min(RU_max_load[1], RU_demand[1]) + RU_0
        RU_consumed_ = min(rar_w_sf[1], RU_consumed_)
        DCI_consumed_ = win_1 * min(DCI_max_load[1], DCI_demand[1]) + DCI_0
        DCI_consumed_ = min(DCIs[1], DCI_consumed_)
        RU_consumed[i] = RU_consumed_
        DCI_consumed[i] = DCI_consumed_
        msg3_1 += win_1 * max_msg3_rate(rar_w_sf[1] - RU_consumed_, RUs[1], DCIs[1] - DCI_consumed_, dci_1)
    msg3_1 = msg3_1/sum(window[1])

    if debug:
        print(RU_consumed)
        print(DCI_consumed)
        print('------------------')

    # CE 2
    msg3_2 = 0
    for RU_1, DCI_1, win_2 in zip(RU_consumed, DCI_consumed, window[2]):
        RU_consumed_ = win_2 * min(RU_max_load[2], RU_demand[2]) + RU_1
        RU_consumed_ = min(rar_w_sf[2], RU_consumed_)
        DCI_consumed_ = win_2 * min(DCI_max_load[2], DCI_demand[2]) + DCI_1
        DCI_consumed_ = min(DCIs[2], DCI_consumed_)
        msg3_2 += win_2 * max_msg3_rate(rar_w_sf[2] - RU_consumed_, RUs[2], DCIs[2] - DCI_consumed_, dci_2)
    msg3_2 = msg3_2/sum(window[2])

    if debug:
        print(RU_consumed)
        print(DCI_consumed)

    return [msg3_0, msg3_1, msg3_2]

# MAIN FUNCTION 

def compute_largest_rate(arrival_rate_per_CE, nsc, periods, msg3_RUs, N_sfs, msg3_detect = [0.95, 0.92, 0.85], debug = False):
    '''
    This function estimates the maximum RA detection rate per CE level
    msg3_err denotes the ratio of successful connections over msg3 messages
    a successful connection imples an msg3 that is actually sent 
    (there are enough opportunities in the RAR window) and detected 
    '''
    max_rates = [0, 0, 0]
    # arrival_rate_plus = [a_*(2 - p_) for a_,p_ in zip(arrival_rate_per_CE, msg3_detect)]
    # arrival_rate_plus = arrival_rate_per_CE

    # set the maximum number of msg3 sent per RAR_window
    k_max, _, _ = RAR_resources_per_window(periods, msg3_RUs, N_sfs)
    msg3_rates = msg3_rate_estimation(arrival_rate_per_CE, periods, msg3_RUs, N_sfs, debug = debug)

    for i, (nsc_i, period_i, rate_i) in enumerate(zip(nsc, periods, arrival_rate_per_CE)):
        nsc_i = 4 * nsc_i # transform into NPRACH subcarriers
        if rate_i == 0.0:
            max_rates[i] = 1.0
            continue
        max_detections = 0
        for n in range(1, nsc_i + 1):
            average_detections_for_n = 0
            for k in range(1, n+1):
                average_detections_for_n += min(k,k_max[i]) * p_detections[(k, n, nsc_i)]
            if average_detections_for_n > max_detections:
                max_detections = average_detections_for_n
        max_rates[i] = msg3_detect[i] * min(msg3_rates[i], max_detections) / period_i

    if debug:
        print(f'max_rates: {max_rates}')
    
    return max_rates

params = power_distribution_parameters([])

# AUXILIARY FUNCTION FOR THE access_flows_estimation FUNCTION
def RAR_parameters(th_C1, th_C0, arrival_rate, probabilities):
    msg3_RUs = compute_msg3_rus(th_C1, th_C0)
    N_sfs = compute_NPRACH_sf(th_C1, th_C0)
    arrival_rate_per_CE = arrival_flows_estimation(th_C1, th_C0, arrival_rate, probabilities, th_values)
    N_sfs = [0 if a == 0 else N_sfs[i] for i, a in enumerate(arrival_rate_per_CE)]
    return msg3_RUs, N_sfs, arrival_rate_per_CE
