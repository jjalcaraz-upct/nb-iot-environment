#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines system's parameters, the system observation, and the control items

Created on Oct 27, 2022

@author: juanjosealcaraz

"""

# global system parameters
N_users = 4 # max number of users reported in the state
Horizon = 40 # length of the carrier observation horizon in subframes
N_carriers = 1 # number of carriers

# auxiliary function to set the global parameters
def set_global_parameters(N = N_users, H = Horizon, Nc = N_carriers):
    global N_users, Horizon, N_carriers
    N_users = N
    Horizon = H
    N_carriers = Nc

# system observation
user_items = {
    'connection_time': 0,
    'loss': 1,
    'sinr': 2,
    'buffer': 3
}

population_items = {
    'total_ues': 0,
    'ues_CE0': 1,
    'ues_CE1': 2,
    'ues_CE2': 3
}

RAR_items = {
    'p_anchor': 0,
    'rar_window': 1
}

NPRACH_items = {
    'period': 0,
    'sc': 1,
    'rep': 2,
    'detections': 3
}

# system control
control_items = { # [0, 0, 2, 0, 4, 2, 1, 2, 4, 2, 2, 12, 2, 0, 0, 1, 2, 2, 2, 3, 1]
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
    'sc_C2': 20
}

N_actions = 21

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
    'Nrep': [7,7],
    'backoff': [12,12],
    'rar_window': [7,7],
    'mac_timer': [7,7],
    'transmax': [10,10],
    'panchor': [15,15],
    'period_C0': [7,7],
    'rep_C0': [7,7],
    'sc_C0': [3,7],
    'period_C1': [7,7],
    'rep_C1': [7,7],
    'sc_C1': [3,7],
    'period_C2': [7,7],
    'rep_C2': [7,7],
    'sc_C2': [3,7]
}

# default action values:
control_default_values = { # [0, 0, 1, 0, 4, 0, 1, 2, 4, 2, 2, 12, 2, 0, 0, 1, 2, 2, 2, 3, 1]
    'carrier': 0,
    'id': 0,
    'Imcs': 2,
    'ce_level': 1,
    'rar_Imcs': 1,
    'N_ru': 4,
    'delay': 0,
    'sc': 3,
    'Nrep': 1,
    'backoff': 2,
    'rar_window': 4,
    'mac_timer': 2,
    'transmax': 2,
    'panchor': 12,
    'period_C0': 2,
    'rep_C0': 0,
    'sc_C0': 0,
    'period_C1': 1,
    'rep_C1': 2,
    'sc_C1': 2,
    'period_C2': 2,
    'rep_C2': 3,
    'sc_C2': 1
}

# CE parameters used by population
SRP = 35.0 # reference signal receive power dBm
CE_thresholds = [-103.0, -85.0] # adopted values
# CE_thresholds = [-105.0, -85.0] # tentative values
# CE_thresholds = [-110.0, -80.0] # tentative values
# CE_thresholds = [-115.0, -85.0] # tentative values

# system state dimension
user_vars = len(user_items) # number of observed variables per UE
population_vars = len(population_items)
rar_vars = len(RAR_items)
nprach_vars = len(NPRACH_items) # number of observed variables per CE level
state_dim = N_users * user_vars + population_vars + rar_vars + nprach_vars * 3 + Horizon * N_carriers

# subsets of state indexes 
carrier_indexes = list(range(state_dim - Horizon * N_carriers, state_dim))
scheduling_indexes = list(range(0, N_users * user_vars + 1)) + carrier_indexes
ce_selection_indexes = list(range(N_users * user_vars + 1, N_users * user_vars + population_vars))
nprach_indexes = list(range(N_users * user_vars + 1, state_dim - Horizon * N_carriers))

# NODE B: NPDCCH parameters
NPDCCH_period = 10 # 20,30 ... length of the NPDCCH period in subframes (NPDCCH is always located in the 1st subframe)
NPDCCH_sf = 4 # number of consecutive NPDCCH subframes 
NPDCCH_sf_per_CE = [1, 2, 4] # NPDCCH repetitions per CE level
CE_for_NPDCCH_sf_left = [0, 0, 1, 1, 2] # max CE level allowed given the NPDCCH sf left

# NODE B: NPRACH parameters
NPRACH_update_period = 100 * NPDCCH_period

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

# TBS_table = [
#     [16, 24, 32, 40, 56, 72, 88, 104, 120, 136, 144, 176, 208, 224], 
#     [32, 56, 72, 104, 120, 144, 176, 224, 256, 296, 328, 376, 440, 488],
#     [56, 88, 144, 176, 208, 224, 256, 328, 392, 456, 504, 584, 680, 744],
#     [88, 144, 176, 208, 256, 328, 392, 472, 536, 616, 680, 776, 1000, 1128],
#     [120, 176, 208, 256, 328, 424, 504, 584, 680, 776, 872, 1000, 1128, 1256],
#     [152, 208, 256, 328, 408, 504, 600, 712, 808, 936, 1000, 1096, 1384, 1544],
#     [208, 256, 328, 440, 552, 680, 808, 1000, 1096, 1256, 1384, 1608, 1800, 2024],
#     [256, 344, 424, 568, 680, 872, 1000, 1224, 1384, 1544, 1736, 2024, 2280, 2536]
#     ] # Transport Block Size table TBS_table[I_ru][I_tbs] (Table 7.18)

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

NPRACH_periodicity_list = [40, 80, 160, 240, 320, 640, 1280, 2560] # in ms (Table 6.3)
NPRACH_N_sc_list = [[12, 24, 36, 48],  # total number of 3.75 subcarriers of a NRACH resource (Table 6.3)
                    [12, 24, 36, 48, 60, 72, 84, 96], # one anchor + one non-anchor carrier
                    [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144] # one anchor + 2 non-anchor carriers
                    ]
N_rep_preamble_list = [1, 2, 4, 8, 16, 32, 64, 128] # number of random access preamble repetitions per attempt of each NPRACH resource (Table 6.3)
probability_anchor = [
                    0, 1/16, 1/15, 1/14, 1/13, 1/12, 1/11, 1/10, 
                    1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2
                    ]

# Funtions used by the Node B to generate the state
# for ue selection
def ue_state(time, selectable_ues, total_ues):
    state = [0] * state_dim
    i = 0
    for ue in selectable_ues:
        state[i] = min(1.0, (time - ue.t_connection)/1000.0) # connection time in sf
        state[i + 1] = min(1.0, max(0.0, (ue.loss - 100)/50)) # loss
        state[i + 2] = 0 # 0 if new data
        state[i + 3] = ue.buffer
        if not ue.new_data:
            sinr = min(max(ue.sinr, -40),30) # [-40, 30]
            state[i + 2] = min(1.0, (40 + sinr)/70) # normalized sinr if HARQ retransmission
        i += user_vars
    state[N_users * user_vars] = min(1.0, total_ues/100) # position 12
    return state

# for CE selection in rar window
def rar_state(ues_per_CE):
    state = [0] * state_dim
    i = N_users * user_vars + 1
    for ce, ues in enumerate(ues_per_CE):
        state[i + ce] = min(1.0, ues / 20.0)
    return state

# for NPRACH configuration
def nprach_state(NPRACH_conf, NPRACH_detections, n_c):
    state = [0] * state_dim
    i = N_users * user_vars + population_vars # 16
    state[i] = NPRACH_conf['p_anchor']
    state[i + 1] = NPRACH_conf['rar_window'] / 7.0
    i = i + rar_vars # 18
    for ce in range(3):
        d = NPRACH_detections[ce]
        mean_d = sum(d) / max(1,len(d))
        period = NPRACH_conf[ce]['periodicity']
        scs = NPRACH_conf[ce]['subcarriers']
        reps = NPRACH_conf[ce]['repetitions']
        N_scs_list = NPRACH_N_sc_list[n_c]
        state[i + ce * nprach_vars] = 1.0 - period / 7.0
        state[i + ce * nprach_vars + 1] = scs / len(N_scs_list)
        state[i + ce * nprach_vars + 2] = reps / 7.0
        state[i + ce * nprach_vars + 3] = min(1.0, mean_d / N_scs_list[scs]) 
    return state