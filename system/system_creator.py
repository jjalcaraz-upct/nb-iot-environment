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