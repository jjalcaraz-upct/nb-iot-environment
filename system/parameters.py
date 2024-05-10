#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines system's parameters, the system observation, and the control items

Created on Oct 27, 2022

@author: juanjosealcaraz

"""
import numpy as np

# global system parameters
N_users = 4 # max number of users reported in the state
Horizon = 40 # length of the carrier observation horizon in subframes
N_carriers = 1 # number of carriers
MAX_BUFFER = 600 # bits
msg3_tbs_size = 88 # bits

# Cell parameters
''' Based on Hwang, Jeng-Kuang, Cheng-Feng Li, and Chingwo Ma. 
"Efficient detection and synchronization of superimposed NB-IoT NPRACH preambles." 
IEEE Internet of Things Journal 6.1 (2018): 1173-1182.'''
N = -121.4 # Noise in dBm
Tx_pw = 23 # dBm
Gmax = 15 #dBi
sigma_F = 8 # dB standard deviation of the log Normal fading
maxRange = 10 # Km (maximum cell range)
L_o = 0 # penetration loss (walls, ground, etc)
F = 3 # Noise Figure in dB
radius = 1/2

max_p = 5 # MAX NPRACH PERIOD FOR RL AGENTS

# CE level configuration values
SRP = 35.0 # reference signal receive power dBm
threshold_list = [-116, -115, -114, -113, -112, -110, -108, -106, -104, -102, -100, -98, 0]

# the last value is used to eliminate levels, e.g.
# CE_thresholds = [-102.5, 0] --> Only CE2 and CE1 (no CE0)
# CE_thresholds = [0, 0] --> Only CE2 (neither CE1 nor CE0)
max_th = len(threshold_list) - 1

thr_to_thrI = {thr: i for i, thr in enumerate(threshold_list)}

th_values = threshold_list[:-1]
th_indexes = {}

values_ = len(th_values)
th_pairs = []
for i, th_1 in enumerate(th_values):
    if i == values_:
        break
    for j in range(i, values_):
        th_0 = th_values[j]
        th_pairs.append((th_1,th_0))
        th_indexes[(th_1,th_0)] = [i, j]

NPDCCH_sf = 8 # 4 number of consecutive NPDCCH subframes 

thr_to_n_rep = { 
    -116: 3, # 8
    -115: 3, # 8
    -114: 2, # 4
    -113: 2, # 4
    -112: 2, # 4
    -111: 1, # 2
    -110: 1, # 2
    -109: 1, # 2
    -108: 1, # 2
}

no_rep_th = -107

# msg3 configuration for th_level computed using just nominal values
msg_3_conf = {
    -116: (2,1,16),
    -115: (2,1,16),
    -114: (2,1,16),
    -113: (2,1,16),
    -112: (2,1,8),
    -111: (2,1,8),
    -110: (2,1,8),
    -108: (2,1,8),
    -107: (2,1,4),
    -106: (2,1,4),
    -105: (2,1,4),
    -104: (2,1,4),
    -103: (2,1,2),
    -102: (2,1,2),
    -101: (2,1,2),
}

no_rep_msg3_th = -100

# auxiliary function to set the global parameters
def set_global_parameters(N_ = N_users, H = Horizon, Nc = N_carriers):
    global N_users, Horizon, N_carriers
    N_users = N_
    Horizon = H
    N_carriers = Nc

# ###################################################
#                  OBSERVATION ITEMS
# ###################################################

obs_dict = { # [length, normalization constant]
    'total_ues': [1, 100],
    'connection_time': [N_users, 1000],
    'loss': [N_users, 50],
    'sinr': [N_users, 70],
    'buffer': [N_users, 100],
    'detection_ratios': [3, 1],
    'colision_ratios': [3, 1],
    'msg3_detection': [3, 1],
    'NPUSCH_occupation': [1, 1],
    'NPRACH_occupation': [1, 1],
    'av_delay': [1, 10000],
    'ues_per_CE': [3, 20],
    'RAR_in': [3, 20],
    'RAR_sent': [3, 20],
    'RAR_detected': [3, 20],
    'RAR_ids': [3, 20],
    'NPDCCH_sf_left': [1, NPDCCH_sf],
    'sc_C0': [1, 3],
    'sc_C1': [1, 3],
    'sc_C2': [1, 3],
    'period_C0': [1, 7],
    'period_C1': [1, 7],
    'period_C2': [1, 7],
    'th_C0': [1, values_],
    'th_C1': [1, values_],
    'distribution': [values_ + 1, 1],
    'carrier_state': [Horizon, 1],
}

NPRACH_items = [
    'backoff',
    'rar_window',
    'mac_timer',
    'transmax',
    'panchor',
    'th_C1',
    'th_C0',
    'period_C0',
    'rep_C0',
    'sc_C0',
    'period_C1',
    'rep_C1',
    'sc_C1',
    'period_C2',
    'rep_C2',
    'sc_C2'
]

# ###################################################
#                  CONTROL ITEMS
# ###################################################

control_items = {
    'carrier': 0,
    'id': 1,
    'Imcs': 2,
    'ce_level': 1, # overlap is ok in different states
    'rar_Imcs': 2, # more overlap
    'N_ru': 3,
    'delay': 4,
    'sc': 5,
    'Nrep': 6,
    'backoff': 7,
    'rar_window': 8,
    'mac_timer': 9,
    'transmax': 10,
    'panchor': 11,
    'period_C0': 12,
    'rep_C0': 13,
    'sc_C0': 14,
    'period_C1': 15,
    'rep_C1': 16,
    'sc_C1': 17,
    'period_C2': 18,
    'rep_C2': 19,
    'sc_C2': 20,
    'th_C1': 21,
    'th_C0': 22
}

N_actions = 23

# max values

control_max_values = {
    'carrier': [0,1],
    'id': [N_users-1, N_users-1],
    'Imcs': [6,6],
    'ce_level': [2,2],
    'rar_Imcs': [2,2],
    'N_ru': [7,7],
    'delay': [3,3],
    'sc': [3,3],
    # 'Nrep': [7,7],
    'Nrep': [4,4],
    'backoff': [12,12],
    'rar_window': [7,7],
    'mac_timer': [7,7],
    'transmax': [10,10],
    'panchor': [15,15],
    'period_C0': [max_p,max_p],
    'rep_C0': [7,7],
    'sc_C0': [3,7],
    'period_C1': [max_p,max_p],
    'rep_C1': [7,7],
    'sc_C1': [3,7],
    'period_C2': [max_p,max_p],
    'rep_C2': [7,7],
    'sc_C2': [3,7],
    'th_C1': [max_th,max_th],
    'th_C0': [max_th,max_th]   
}

# default action values:
control_default_values = {
    'carrier': 0,
    'id': 0,
    'Imcs': 2,
    'ce_level': 1,
    'rar_Imcs': 1,
    'N_ru': 4,
    'delay': 0,
    'sc': 3,
    'Nrep': 1,
    'backoff': 2, # should be larger than the largest NPRACH period
    'rar_window': 6,
    'mac_timer': 2,
    'transmax': 4,
    'panchor': 12,
    'period_C0': 3, # 3,
    'rep_C0': 0,
    'sc_C0': 1,
    'period_C1': 2, # 2,
    'rep_C1': 0,
    'sc_C1': 1, 
    'period_C2': 3, # 3,
    'rep_C2': 2,
    'sc_C2': 1,
    'th_C1': 5,
    'th_C0': 9 # max_th # 0 --> No CE0
}

# NODE B: NPDCCH parameters

NPDCCH_period = 20 # 30, 10 ... length of the NPDCCH period in subframes (NPDCCH is always located in the 1st subframe)
NPDCCH_sf_per_CE = [1, 2, 4] # NPDCCH repetitions per CE level
CE_for_NPDCCH_sf_left = [0, 0, 1, 1, 2, 2, 2, 2, 2] # max CE level allowed given the NPDCCH sf left

# NODE B: NPRACH parameters

NPRACH_update_period = 100 * NPDCCH_period #

# NODE B: auxiliary dictionary for automatic selection of sc 
N_sc_selection_list = {
    12: [6, 3, 1],
    6: [12, 3, 1],
    3: [12, 6, 1],
    1: [12, 6, 3]
    }

# NODE B: auxiliary dictionary to determine the number of sf per ru
N_sf_per_ru = {
    12: 1,
    6: 2,
    3: 4,
    1: 8
}

# Imcs mapping to Itbs, IN_ru
Imcs_to_Itbs = [0,1,2,3,4,6,8]
Itbs_to_Imcs = {0:0, 1:1, 2:2, 3:3, 4:4, 6:5, 8:6}
Imcs_to_N_ru = [10,8,6,5,4,3,2]

# Imcs mapping to N_ru in RAR
RAR_Imcs_to_N_ru = [4,3,1]

# NPUSCH parameters
N_rep_list = [1, 2, 4, 8, 16, 32, 64, 128] # repetitions (Table 7.13)
N_ru_list = [1, 2, 3, 4, 5, 6, 8, 10] # resource units (Table 7.14)
sf_delay_list = [8, 16, 32, 64] # delay in subframes (Table 7.15)
N_sc_list = [1, 3, 6, 12] # number of subcarriers per RU (Table 7.16)
N_sf_list = [8, 4, 2, 1] # number of sf per RU (Table 7.29) Warning: 2 slots = 1 sf


TBS_table = [[16, 32, 56, 88, 120, 152, 208, 256],
             [24, 56, 88, 144, 176, 208, 256, 344],
             [32, 72, 144, 176, 208, 256, 328, 424],
             [40, 104, 176, 208, 256, 328, 440, 568],
             [56, 120, 208, 256, 328, 408, 552, 680],
             [88, 176, 256, 392, 504, 600, 808, 1000],
             [120, 256, 392, 536, 680, 808, 1096, 1384]
]

TBS_list = [e for lista in TBS_table for e in lista]
TBS_list = sorted(list(set(TBS_list)))

tbs_lists = {0: [16, 32, 56, 88, 120, 152, 208, 256],
             1: [24, 56, 88, 144, 176, 208, 256, 344],
             2: [32, 72, 144, 176, 208, 256, 328, 424],
             3: [40, 104, 176, 208, 256, 328, 440, 568],
             4: [56, 120, 208, 256, 328, 408, 552, 680],
             6: [88, 176, 256, 392, 504, 600, 808, 1000],
             8: [120, 256, 392, 536, 680, 808, 1096, 1384]
            } # mapping from I_tbs to TBS

# 'backoff': 2, -> 512 ms
# 'rar_window': 4, -> 6 NPDCCH periods
# 'mac_timer': 2, -> 3 NPDCCH periods
# 'transmax': 2, -> 5 preamble transmission

# Contention parameters
backoff_list = [0, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288] # max backoff values in ms (Table 6.6)

# NPRACH parameters
RAR_WindowSize_list = [2,3,4,5,6,7,8,10] # duration of RAR window in NPDCCH periods (Table 6.2)
MAC_timer_list = [1,2,3,4,8,16,32,64] # duration of contention resolution timer in NPDCCH periods (Table 6.2)
preamble_trans_max_CE_list = [3,4,5,6,7,8,10,20,50,100,200] # maximum number of preamble transmissions before moving to an upper CE level (Table 6.2)

# NPRACH_periodicity_list = [40, 80, 160, 240, 320, 640, 1280, 2560] # in ms (Table 6.3)
NPRACH_periodicity_list = [80, 160, 240, 320, 640, 1280, 2560] # in ms (Table 6.3)
NPRACH_N_sc_list = [[12, 24, 36, 48],  # total number of 3.75 subcarriers of a NRACH resource (Table 6.3)
                    [12, 24, 36, 48, 60, 72, 84, 96], # one anchor + one non-anchor carrier
                    [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144] # one anchor + 2 non-anchor carriers
                    ]
N_rep_preamble_list = [1, 2, 4, 8, 16, 32, 64, 128] # number of random access preamble repetitions per attempt of each NPRACH resource (Table 6.3)
probability_anchor = [
                    0, 1/16, 1/15, 1/14, 1/13, 1/12, 1/11, 1/10, 
                    1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2
                    ]

# configurable NPRACH parameters
top_i = max_p + 1
period_list = NPRACH_periodicity_list[:top_i] # [80, 160, 240, 320, 640, 1280, 2560] # in ms (Table 6.3)
N_sc_list = [nsc_//4 for nsc_ in NPRACH_N_sc_list[0]] # [12, 24, 36, 48] --> [3, 6, 9, 12]

# number of subrames of each NPRACH
NPRACH_N_sf_list = [int(np.ceil(N_rep * 5.6)) for N_rep in N_rep_preamble_list]

def compute_NPRACH_sf(th_C1, th_C0):
    '''
    Computes the number of subframes of the NPRACH for each CE level 
    based on the RSRP thresholds that determine the CE levels
    '''
    min_th = threshold_list[0]
    rep_C2 = thr_to_n_rep[min_th]
    sf_c2 = NPRACH_N_sf_list[rep_C2]
    if th_C1 < no_rep_th:
        th_C1 = max(min_th, th_C1)
        rep_C1 = thr_to_n_rep[th_C1]
    else:
        rep_C1 = 0
    sf_c1 = NPRACH_N_sf_list[rep_C1]
    if th_C0 < no_rep_th:
        th_C0 = max(min_th, th_C0)
        rep_C0 = thr_to_n_rep[th_C0]
    else:
        rep_C0 = 0
    sf_c0 = NPRACH_N_sf_list[rep_C0]
    if th_C0 == 0 or th_C1 == th_C0: # only two CE levels: CE1 and CE2
        sf_c0 = 0
    return [sf_c0, sf_c1, sf_c2]