#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for simulation and presentation of results.

Created on May 2022

@author: juanjosealcaraz

"""

import numpy as np
from scipy import stats
from node_b import NodeB
from population import Population
from carrier import Carrier
from channel import Channel
from access_procedure import AccessProcedure
from rx_procedure import RxProcedure
from ue_generator import UEGenerator
from perf_monitor import PerfMonitor
from message_switch import MessageSwitch
from parameters import control_items, control_max_values, control_default_values, state_dim, N_carriers, Horizon

max_values = [v[N_carriers-1] for k,v in control_max_values.items() if k not in ['ce_level', 'rar_Imcs']]
min_values = state_dim*[0]

action_basic = [v for k,v in control_default_values.items() if k not in ['ce_level', 'rar_Imcs']]

def generate_random_action(rng):
    return rng.integers(low = min_values, high = max_values)

carrier_i = control_items['carrier']
id_i = control_items['id']
sc_i = control_items['sc']

def generate_reasonable_action(rng, o):
    action = action_basic
    if N_carriers > 1:
        c0 = sum(o[range(state_dim - 2*Horizon, state_dim - Horizon)])
        c1 = sum(o[range(state_dim - Horizon, state_dim)])
        if c1 < c0:
            action[carrier_i] = 1
    action[id_i] = 0
    action[sc_i] = rng.integers(4) # subcarriers
    return action

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def create_system(rng, conf):
    # get the parameters
    ANIMATE_CARRIER = conf['animate_carrier']
    STATISTICS = conf['statistics']
    ANIMATE_STATS = conf['animate_stats']
    ratio = conf['ratio']
    M = conf['M']
    levels = conf['levels']
    reward_criteria = conf['reward_criteria']
    sc_adjustment = conf['sc_adjustment']
    mcs_automatic = conf['mcs_automatic']
    buffer_range = conf['buffer_range']
    sort_criterium = conf.get('sort_criterium', 't_connection')

    print(f' > CREATE SYSTEM: sort criterium = {sort_criterium}')

    # create the system 
    # each element must be created individually
    # and each element subscribes itself to the events it is in charge of processing
    m = MessageSwitch()
    carrier = Carrier(m, animation = ANIMATE_CARRIER)
    channel = Channel(rng, m)
    ue_generator = UEGenerator(rng, m, M = M, ratio = ratio, buffer_range = buffer_range)
    population = Population(rng, m, levels = levels)
    access = AccessProcedure(rng,m)
    receptor = RxProcedure(m)
    perf_monitor = PerfMonitor(m, reward_criteria = reward_criteria, statistics = STATISTICS, animation = ANIMATE_STATS)
    node = NodeB(m, sc_adjustment = sc_adjustment, mcs_automatic = mcs_automatic, sort_criterium = sort_criterium)

    return node, perf_monitor, population, carrier