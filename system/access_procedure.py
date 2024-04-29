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
        self.collided_preambles_per_CE = [0, 0, 0]
        self.detection_history = [[], [], []]
        self.collision_history = [[], [], []]
        self.contenders_per_CE = [0, 0, 0]
        self.RAR_counter = [0, 0, 0] # RAR window counter

    def NPRACH_start(self, event):
        '''
        Implements the Random Access Procedure in a NPRACH resource
        Each UE selects a carrier (if there are more than 1) and a preamble
        Then, the channel model determines which ones are detected
        The channel updates the UE states to RAR, COLLIDED or CAPTURE
        '''
        t = event.t
        CE_level = event.CE_level
        N_pream = event.N_pream
        N_sc = event.N_sc
        N_sf = event.N_sf
        N_rep = event.N_rep
        counter = self.RAR_counter[CE_level]
        p = self.probability_anchor
        self.m.perf_monitor.register_NPRACH_resources(N_sc, N_sf) # register resource use
        ra_ues = self.m.population.NPRACH_start(CE_level, t, counter) # get the list of ues in random access state
        n_ues = len(ra_ues) # number of ra ues
        self.contenders_per_CE[CE_level] = n_ues

        if DEBUG:
            print(f'    {t}: NPRACH CE{CE_level} starts ')

        if n_ues > 0:
            na_ues = 0 # non-anchor ues
            Na_sc = self.m.carrier_set.check_sc(t, CE_level) # number of subcarriers in the non-anchor carrier
            preambles = []
            if Na_sc:
                na_ues = self.rng.binomial(n_ues, p)
                # the preambles in the non-anchor carrier are numbered with larger integers
                preambles.extend(self.rng.integers(low = N_pream, high = N_pream + 4 * Na_sc, size = na_ues))
                self.m.perf_monitor.register_NPRACH_resources(Na_sc, N_sf) # register resource use
            a_ues = n_ues - na_ues # anchor ues
            preambles.extend(self.rng.integers(N_pream, size = a_ues))
            
            outcome = {}
            for preamble, ue in zip(preambles, ra_ues):
                ue.preamble = preamble
                outcome.setdefault(preamble, []).append(ue)

            detected_preambles = 0
            detected_signals = 0
            for ues in outcome.values():
                preable_detected, signal_detected = self.m.channel.preamble_detection(ues, N_rep) # check reception of preambles
                detected_preambles += preable_detected
                detected_signals += signal_detected

            # the number of detected preambles determines the number of msg3 messages sent by the node
            collided_preambles = detected_signals - detected_preambles
            self.detected_preambles_per_CE[CE_level] = detected_preambles
            self.collided_preambles_per_CE[CE_level] = collided_preambles
            self.detection_history[CE_level].append(detected_preambles) # ratio
            self.collision_history[CE_level].append(collided_preambles) # ratio

        # Note: the carrier schedules the NPRACH_end event

    def NPRACH_end(self, event):
        '''
        Invokes the method start_RAR_window of the Node B 
        informs the node B of the number of detected preambles in the CE_level
        '''
        CE_level = event.CE_level
        t = event.t
        counter = self.RAR_counter[CE_level]

        # nprach performance samples
        n_ues = self.contenders_per_CE[CE_level]
        detected_preambles = self.detected_preambles_per_CE[CE_level] 
        self.m.perf_monitor.nprach_sample(n_ues, detected_preambles, CE_level, t)

        if DEBUG:
            print(f'    {event.t}: NPRACH CE{CE_level} ends')
        if n_ues > 0: # if no contenders no action required
            self.m.node.start_RAR_window(detected_preambles, CE_level, t, counter)
            # now reset
            self.detected_preambles_per_CE[CE_level] = 0
            self.collided_preambles_per_CE[CE_level] = 0
            self.contenders_per_CE[CE_level] = 0
        
        self.RAR_counter[CE_level] += 1
    
    def get_histories(self):
        return self.detection_history, self.collision_history
    
    def reset_history(self):
        self.detection_history = [[], [], []]
        self.collision_history = [[], [], []]   