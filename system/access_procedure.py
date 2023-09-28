#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2022

@author: juanjosealcaraz
"""

from .event_manager import subscribe

DEBUG = False

class AccessProcedure:
    '''
    Implements the access protocol for the UEs.
    This class communicates with the population and with the carrier.
    Handled events: NPRACH_start and NPRACH_end. 
    '''
    def __init__(self, rng, m):
        self.rng = rng
        self.m = m
        self.m.set_access_procedure(self)
        self.reset()

        subscribe('NPRACH_start', self.NPRACH_start)
        subscribe('NPRACH_end', self.NPRACH_end)

    def reset(self):
        self.probability_anchor = 0
        self.detected_preambles_per_CE = [0, 0, 0]
        self.contenders_per_CE = [0, 0, 0]

    def NPRACH_start(self, event):
        '''
        Implements the Random Access Procedure in a NPRACH resource
        Each UE selects a carrier (if there are more than 1) and a preamble
        Then, the channel model determines which ones are detected
        The channel updates the UE states to RAR, COLLIDED or CAPTURE
        '''
        t = event.t
        CE_level = event.CE_level
        N_sc = event.N_sc
        N_rep = event.N_rep
        p = self.probability_anchor
        ra_ues = self.m.population.NPRACH_start(CE_level, t) # get the list of ues in random access state
        a_ues = len(ra_ues) # anchor ues
        self.contenders_per_CE[CE_level] = a_ues

        if a_ues > 0:
            na_ues = 0 # non-anchor ues
            Na_sc = self.m.carrier_set.check_sc(t, CE_level) # number of subcarriers in the non-anchor carrier
            preambles = []
            if Na_sc:
                na_ues = self.rng.binomial(a_ues, p)
                a_ues -= na_ues
                # the preambles in the non-anchor carrier are numbered with larger integers
                preambles.extend(self.rng.integers(low = N_sc, high = N_sc + Na_sc, size = na_ues))
            preambles.extend(self.rng.integers(N_sc, size = a_ues))
            
            outcome = {}
            for preamble, ue in zip(preambles, ra_ues):
                ue.preamble = preamble
                outcome.setdefault(preamble, []).append(ue)

            detected_preambles = 0
            for ues in outcome.values():
                detected_preambles += self.m.channel.preamble_detection(ues, N_rep) # check reception of preambles

            # the number of detected preambles determines the number of msg3 messages sent by the node
            self.detected_preambles_per_CE[CE_level] = detected_preambles

            # nprach performance samples
            self.m.perf_monitor.nprach_sample(a_ues, CE_level, arrival = True)
            self.m.perf_monitor.nprach_sample(detected_preambles, CE_level)

        # Note: the carrier schedules the NPRACH_end event

    def NPRACH_end(self, event):
        '''
        Invokes the method start_RAR_window of the Node B 
        informs the node B of the number of detected preambles in the CE_level
        '''
        CE_level = event.CE_level
        if self.contenders_per_CE[CE_level] > 0: # if no contenders no action required
            detected_preambles = self.detected_preambles_per_CE[CE_level]
            self.m.node.start_RAR_window(detected_preambles, CE_level, event.t)
            self.detected_preambles_per_CE[CE_level] = 0 # reset this value
            self.contenders_per_CE[CE_level] = 0 # reset this value 