#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function that creates the system with all the elements

Created on May 2022

@author: juanjosealcaraz

"""

from .node_b import NodeB
from .population import Population
from .carrier import Carrier
from .channel import Channel
from .access_procedure import AccessProcedure
from .rx_procedure import RxProcedure
from .ue_generator import UEGenerator
from .perf_monitor import PerfMonitor
from .message_switch import MessageSwitch

def create_system(rng, conf):
    # get the parameters
    ratio = conf['ratio']
    M = conf['M']
    buffer_range = conf['buffer_range']
    reward_criteria = conf['reward_criteria']
    sort_criterium = conf.get('sort_criterium', 't_connection')
    levels = conf.get('levels', [0,1,2])
    max_d = conf.get('max_d', None)
    sc_adjustment = conf.get('sc_adjustment', True)
    mcs_automatic = conf.get('mcs_automatic', True)
    ce_mcs_automatic = conf.get('ce_mcs_automatic', True)
    tx_all_buffer = conf.get('tx_all_buffer', True)
    ANIMATE_CARRIER = conf.get('animate_carrier', False)
    STATISTICS = conf.get('statistics', False)
    ANIMATE_STATS = conf.get('animate_stats', False)
    TRACES = conf.get('traces', False)
    RESOURCE_MONITOR = conf.get('resource_monitor', False)
    RANDOM = conf.get('random', False)
    clustered = conf.get('clustered', False)
    cluster_ues = conf.get('cluster_ues', [10, 30])
    cluster_prob = conf.get('cluster_prob', 0.5)
    cluster_range = conf.get('cluster_range', 0.5)

    # create the system 
    # each element must be created individually
    # and each element subscribes itself to the events that it handles
    m = MessageSwitch()
    carrier = Carrier(m, animation = ANIMATE_CARRIER)
    channel = Channel(rng, m, max_d = max_d)
    ue_generator = UEGenerator(rng, m, M = M, ratio = ratio, buffer_range = buffer_range, random = RANDOM)
    population = Population(rng, m, levels = levels, clustered = clustered, cluster_ues = cluster_ues, cluster_prob = cluster_prob, cluster_range = cluster_range)
    access = AccessProcedure(rng,m)
    receptor = RxProcedure(m)
    perf_monitor = PerfMonitor(m, reward_criteria = reward_criteria, statistics = STATISTICS, traces = TRACES, animation = ANIMATE_STATS, resource_monitoring = RESOURCE_MONITOR)
    node = NodeB(m, sc_adjustment = sc_adjustment, mcs_automatic = mcs_automatic, ce_mcs_automatic = ce_mcs_automatic, sort_criterium = sort_criterium, tx_all_buffer = tx_all_buffer)

    return node, perf_monitor, population, carrier