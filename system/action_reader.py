#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines an auxiliary class to extract information from the control action

Created on Jan 19, 2022

@author: juanjosealcaraz

"""

from . import parameters as par

def get_value(action, item):
    i_ = par.control_items[item]
    return action[i_]

class ActionReader:
    '''
    This class is used by the node b to extract the information contained in the control action 
    '''
    def __init__(self):
        pass

    def scheduling_action(self, action, RAR_action = False):
        '''
        Generates the parameters of the allocated carrier resources for NPUSCH or msg3 messages (if RAR_action = True).
        '''
        carrier = get_value(action, 'carrier')

        i = get_value(action, 'id')
        
        N_ru = get_value(action, 'N_ru')

        # get N_rep
        I_rep = get_value(action, 'Nrep')
        N_rep = par.N_rep_list[I_rep]

        # get I_tbs
        I_mcs = get_value(action, 'Imcs')
        if RAR_action:
            I_tbs = min(I_mcs, 2)         
            N_ru = par.RAR_Imcs_to_N_ru[I_mcs] # number of RUs
        else:
            I_tbs = par.Imcs_to_Itbs[I_mcs]

        # delay and subcarriers
        I_delay = get_value(action, 'delay')
        delay = par.sf_delay_list[I_delay] # delay in subframes
        
        I_sc = get_value(action, 'sc')
        N_sc = par.N_sc_list[I_sc] # number of subcarriers

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
        '''
        Extracts the backoff value
        '''
        I_backoff = get_value(action, 'backoff')
        backoff = par.backoff_list[I_backoff]
        return backoff

    def NPRACH_update_action(self, action):
        '''
        NPRACH configuration update.
        '''
        NPRACH_configuration = {}

        for item in par.NPRACH_items:
            NPRACH_configuration[item] = get_value(action, item)

        return NPRACH_configuration