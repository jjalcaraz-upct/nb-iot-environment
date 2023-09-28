#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines an auxiliary class to extract information from the control action

Created on Jan 19, 2022

@author: juanjosealcaraz

"""

from .parameters import *

# required indexes 
carrier_i = control_items['carrier']
id_i = control_items['id']
Imcs_i = control_items['Imcs']
N_ru_i = control_items['N_ru']
delay_i = control_items['delay']
sc_i = control_items['sc']
Nrep_i = control_items['Nrep']
backoff_i = control_items['backoff']
rar_window_i = control_items['rar_window']
mac_timer_i = control_items['mac_timer']
transmax_i = control_items['transmax']
panchor_i = control_items['panchor']
period_i = control_items['period_C0']
rep_i = control_items['rep_C0']
scc_i = control_items['sc_C0']

class ActionReader:
    '''
    This class is used by the node b to extract the information contained in the control action 
    '''
    def __init__(self, NPDCCH_period):
        self.NPDCCH_period = NPDCCH_period

    def scheduling_action(self, action, RAR_action = False):
        '''
        Generates the parameters of the allocated carrier resources for NPUSCH or msg3 messages (if RAR_action = True).
        '''

        carrier = action[carrier_i]
        i = action[id_i]
        I_mcs = action[Imcs_i]
        I_delay = action[delay_i]
        I_sc = action[sc_i]
        I_rep = action[Nrep_i]
        N_ru = action[N_ru_i]

        # get N_rep
        N_rep = N_rep_list[I_rep]

        # get I_tbs
        if RAR_action:
            I_tbs = min(I_mcs, 2)         
            N_ru = RAR_Imcs_to_N_ru[I_mcs] # number of RUs
        else:
            I_tbs = Imcs_to_Itbs[I_mcs]

        # delay and subcarriers
        delay = sf_delay_list[I_delay] # delay in subframes
        N_sc = N_sc_list[I_sc] # number of subcarriers

        params = {
            'carrier': carrier,
            'i': i,
            'I_tbs': I_tbs,
            'delay': delay,
            'N_sc': N_sc,
            'N_rep': N_rep,
            'N_ru': N_ru,
        }

        return params

    def UE_contention_resolution(self, action):
        '''Extracts the backoff value'''
        backoff = backoff_list[action[backoff_i]]
        return backoff

    def NPRACH_update_action(self, action):
        '''
        Generates the parameter NPRACH configuration update.
        It extracts the CE_level, the MCS and the number of repetitions.
        '''
        RAR_WindowSize = RAR_WindowSize_list[action[rar_window_i]] * self.NPDCCH_period 

        UE_kwargs = {
            'MAC_timer': MAC_timer_list[action[mac_timer_i]] * self.NPDCCH_period, 
            'preamble_trans_max_CE': preamble_trans_max_CE_list[action[transmax_i]],
            'probability_anchor': probability_anchor[action[panchor_i]]
            }
        
        CE_args_list = []

        for CE_level in range(3):
            CE_args = [
                action[period_i + CE_level * 3], # NPRACH_periodicity
                action[rep_i + CE_level * 3], # N_rep_preamble
                action[scc_i + CE_level * 3] # NPRACH_N_sc
            ]
            CE_args_list.append(CE_args)
        
        I_rar_window = action[rar_window_i]

        return RAR_WindowSize, UE_kwargs, CE_args_list, I_rar_window