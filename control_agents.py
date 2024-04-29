#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Definition of several types of control agents

Created on Oct 2022

@author: juanjosealcaraz

"""
# import time
# import numpy as np
# from itertools import product
# import matplotlib.pyplot as plt
from wrappers import to_discrete
import system.parameters as par
    
# configuration of the agents that control the system
# the names and indexes of the state and control elements
# are defined in parameter.py

nprach_actions = ['rar_window', 'mac_timer', 'transmax', 'panchor', 
                    'period_C0', 'rep_C0', 'sc_C0',
                    'period_C1', 'rep_C1', 'sc_C1',
                    'period_C2', 'rep_C2', 'sc_C2',
                    'th_C1', 'th_C0']

agents_conf = [
    {'id': 0, # UE selection
    'action_items': ['id'], # action items controlled by this agent
    'obs_items': [],
    'next': 1, # next agent operating in the same nodeb state
    'states': ['Scheduling'] # nodeb state where this agent operates 
    },
    
    {'id': 1, # Imcs, N_rep, N_ru selection
    'action_items': ['Imcs', 'Nrep', 'N_ru'],
    'obs_items': [],
    'next': 2,
    'states': ['Scheduling']
    },

    {'id': 2, # carrier, delay and subcarriers
    'action_items': ['carrier', 'delay', 'sc'],
    'obs_items': [],
    'next': -1,
    'states': ['Scheduling']
    },

    {'id': 3, # ce_level selection
    'action_items': ['carrier', 'ce_level', 'rar_Imcs', 'delay', 'sc', 'Nrep'],
    'obs_items': [],
    'next': -1,
    'states': ['RAR_window']
    },

    {'id': 4, # backoff selection
    'action_items': ['backoff'],
    'obs_items': [],
    'next': -1,
    'states': ['RAR_window_end'],
    },

    {'id': 5, # NPRACH configuration
    'action_items': nprach_actions,
    'obs_items': [],
    'next': -1,
    'states': ['NPRACH_update']
    }
]

# auxiliary functions for retrieving action indexes and max action values
def get_control_indexes(name_list):
    return [par.control_items[name] for name in name_list]

def get_max_control_values(name_list):
    return [par.control_max_values[name][par.N_carriers - 1] for name in name_list]

def get_control_default_values(name_list):
    return [par.control_default_values[name] for name in name_list]

class DummyAgent:
    '''dummy agent that simply applies a fixed action'''
    def __init__(self, dict):
        self.__dict__.update(dict)
        action_items = self.action_items
        self.a_mask = get_control_indexes(action_items)
        self.a_max = get_max_control_values(action_items)
        self.fixed_action = get_control_default_values(action_items)
        self.total_steps = 0

    def reset(self):
        action_items = self.action_items
        self.a_mask = get_control_indexes(action_items)
        self.a_max = get_max_control_values(action_items)
        self.fixed_action = get_control_default_values(action_items)

    def set_action(self, fixed_action):
        self.fixed_action = fixed_action

    def get_action(self, obs, r, info, action):
        # action contains the action so far
        # agents communicate using this argument
        self.total_steps += 1
        return self.fixed_action

    def print_action(self):
        for name, value in zip(self.action_items, self.fixed_action):
            print(f' {name}: {value}')
        print('')


class TrainedAgent(DummyAgent):
    def __init__(self, dict, model, deterministic = False):
        super().__init__(dict)
        self.model = model
        self.deterministic = deterministic

    def get_action(self, obs, r, info, action):
        action, _ = self.model.predict(obs, deterministic = self.deterministic)
        return action


class DiscreteTrainedAgent(TrainedAgent):
    '''
    auxiliary class that encapsulates an rl agent that selects discrete actions but interacts with an envirominent with multidiscrete actions
    '''
    def __init__(self, dict, model, nvec, deterministic = False):
        super().__init__(dict, model, deterministic)
        self.actions = to_discrete(nvec)
    
    def get_action(self, obs, r, info, action):
        action, _ = self.model.predict(obs, deterministic = self.deterministic)
        return self.actions[action]


class RandomUserAgent(DummyAgent):
    '''agent that selects a user at random'''
    def __init__(self, dict, rng):
        super().__init__(dict)
        self.rng = rng
    
    def get_action(self, obs, r, info, action):
        users = min(len(info['ues']), par.N_users)
        if users == 0:
            return [0]
        selection = self.rng.integers(users)
        return [selection]
