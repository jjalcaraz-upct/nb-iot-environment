#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implements an pseudo broker class that allows the objects to communicate with each other

Created on Feb 19, 2022

@author:juanjosealcaraz

"""

class MessageSwitch:
    def __init__(self, node = None, population = None, carrier_set = None, channel = None, access_procedure = None, rx_procedure = None, ue_generator = None, perf_monitor = None):
        self.node = node
        self.population = population
        self.carrier_set = carrier_set
        self.channel = channel
        self.access_procedure = access_procedure
        self.rx_procedure = rx_procedure
        self.ue_generator = ue_generator
        self.perf_monitor = perf_monitor
    
    def set_node(self, node):
        self.node = node
    
    def set_population(self, population):
        self.population = population
    
    def set_carrier_set(self, carrier_set):
        self.carrier_set = carrier_set
    
    def set_channel(self, channel):
        self.channel = channel
    
    def set_access_procedure(self, access_procedure):
        self.access_procedure = access_procedure

    def set_rx_procedure(self, rx_procedure):
        self.rx_procedure = rx_procedure

    def set_ue_generator(self, ue_generator):
        self.ue_generator = ue_generator

    def set_perf_monitor(self, perf_monitor):
        self.perf_monitor = perf_monitor

    def reset(self):
        '''method called by the node_b: reset any entity that needs it'''
        self.population.reset()
        self.carrier_set.reset()
        self.perf_monitor.reset()
        self.ue_generator.reset()
        self.access_procedure.reset()
