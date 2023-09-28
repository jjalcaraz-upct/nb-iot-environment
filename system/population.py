#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implements the Population class that holds the users during their access procedure.
Once they are connected (hve received an msg4 message) they are handled by the node b.

Created on Jan 24, 2022

@author: juanjosealcaraz

"""

from .user import STATE, UE, UE_states
from .event_manager import Event, subscribe, schedule_event
from .parameters import CE_thresholds, SRP

# # UE STATES
# RAO = 0 # waiting for a random access opportunity
# BACKOFF = 1 # in a backoff period
# RA = 2 # in random access
# RAR = 3 # in RAR window (waiting for msg2 = UL grant for msg3)
# CAPTURE = 4 # preamble collision but the preamble was decoded by NB
# COLLIDED = 5 # preamble collision impossible to decode
# CONREQ = 6 # in connection request (waiting for msg4)
# CONNECTED = 7 # msg4 received (needs an UL grant for a scheduling request)
# DATA = 8 # in data transmission (waiting for an UL grant for data) 


class Population:
    '''
    Creates, stores and removes UEs from the simulation.
    Population sends messages to ue_generator and the channel.
    Handled events: Backoff_end, MAC_timer
    Scheduled events: Msg3, Backoff_end, MAC_timer
    '''
    def __init__(self, rng, m, levels = [0,1,2]):
        self.rng = rng
        self.m = m
        self.levels = levels # levels considered in the simulation
        self.m.set_population(self)

        self.reset()

        # random access parameters
        self.preamble_trans_max = 2
        self.MAC_timer = 64

        # event subscription
        subscribe('Backoff_end', self.backoff_end)
        subscribe('MAC_timer', self.MAC_timeout)

    def reset(self):
        # UE storage data structures
        self.ra_lists = [[], [], []] # one ra_list for each CE_level
        self.contention_ues = {} # dictionary of UEs in contention resolution

        # ue counter
        self.ue_count = 0


    ############## UE generation methods ################ 

    def get_new_arrivals(self, t):
        '''
        processes UE_arrival events
        '''
        arrivals, beta_arrivals = self.m.ue_generator.generate_arrivals(t)
        # print(f't:{t} POPULATION generates {arrivals} arrivals')
        arrivals += beta_arrivals
        for _ in range(arrivals):
            self.ue_count += 1
            buffer = self.m.ue_generator.generate_buffer()
            ue = UE(self.ue_count, t_arrival = t, buffer = buffer)
            if beta_arrivals > 0:
                ue.beta = True
                beta_arrivals -= 1
            CE_level = -1
            while CE_level not in self.levels:
                self.m.channel.set_xy_loss(ue)
                P_RSRP = SRP - ue.loss
                if P_RSRP < CE_thresholds[0]:
                    CE_level = 2
                elif P_RSRP < CE_thresholds[1]:
                    CE_level = 1
                else:
                    CE_level = 0
            ue.CE_level = CE_level
            self.ra_lists[ue.CE_level].append(ue)
            self.m.perf_monitor.arrival(ue.CE_level)


    def departure(self, ue):
        '''
        method used by Node B to inform of a UE departure
        '''
        self.m.ue_generator.departure(ue)  

    ############## Event processing methods ################ 
    
    def backoff_end(self, event):
        ue = event.ue
        ue.state = STATE.RAO
        ue.backoff_counter += 1
        del event

    def MAC_timeout(self, event):
        '''
        processes MAC_timer events. The event contains a ue_id_list with 
        the ids of the UEs whose timer started simultaneously
        '''
        ue_id_contention = self.contention_ues.keys()
        timeout_ue_ids = [u_i for u_i in event.ue_id_list if u_i in ue_id_contention]
        for ue_id in timeout_ue_ids:
            ue = self.contention_ues.pop(ue_id)
            ue.state = STATE.RAO
            ue.timeout_counter += 1
            self.ra_lists[ue.CE_level].append(ue)

    
    ############## methods invoked by AccessProcedure ################

    def NPRACH_start(self, CE_level, t):
        '''
        method used by AccessProcedure upon NPRACH_start to get UEs in RAO state 
        and put them in RA state
        '''
        if self.m.ue_generator.t < t:
            self.get_new_arrivals(t)
        RAO_users = (ue for ue in self.ra_lists[CE_level] if ue.state == STATE.RAO)
        ra_ues = []
        for ue in RAO_users:
            ue.state = STATE.RA
            ra_ues.append(ue)
        return ra_ues
    
    ############## methods invoked by NodeB ################

    def msg4(self, ue_id):
        '''
        method used by the Node B to notify the outcome of a msg3 transmission.
        '''
        self.contention_ues.pop(ue_id)

    def msg3_grant(self, CE_level, t, t_msg3_end, I_tbs = 1, N_rep = 2): # msg3_grant(CE_level, reference_time, t_ul_end, I_tbs = I_tbs, N_rep = N_rep)
        '''
        method used by NodeB to allocate resources for a msg3 UL transmission 
        '''
        contention_list = []
        preamble = -1

        def move_to_contention(ue):
            ue.I_tbs = I_tbs
            ue.N_rep = N_rep
            ue.t_contention = t
            contention_list.append(ue)
    
        RAR_ues = (ue for ue in self.ra_lists[CE_level] if ue.state in [STATE.CAPTURE, STATE.RAR])
        
        for i, ue in enumerate(RAR_ues):
            if i == 0:
                preamble = ue.preamble              
                move_to_contention(ue)
                if ue.state == STATE.RAR: # this guy did not collide
                    ue.state = STATE.CONREQ
                    break # contention_list only contains one ue
            elif ue.preamble == preamble:
                move_to_contention(ue) # contention_list contains 2+ ues
        
        for ue in contention_list:
            self.ra_lists[CE_level].remove(ue) # no longer in random access
            # print(f' -- ue id: {ue.id} removed from random access ')
            self.contention_ues[ue.id] = ue # now in contention resolution (connection request)
            # print(f' -- contention list contains ue ids: {self.contention_ues.keys()}')
        
        # (if contention list has any element, which it should)
        # create msg3 and mac_timer events and schedule them
        contenders = len(contention_list)
        if contenders > 0:
            msg3_event = Event('Msg3', ue_list = contention_list)
            schedule_event(t_msg3_end, msg3_event) # schedule msg3
            timer_event = Event('MAC_timer', ue_id_list = [ue.id for ue in contention_list])
            t_exp = max(t + self.MAC_timer, t_msg3_end + 1)
            schedule_event(t_exp, timer_event) # schedule timer

    def RAR_window_end(self, t, CE_level, backoff):
        '''
        method invoked by NodeB when the RAR window of a specific CE_level has finished.
        Notifies the max backoff value, so that the UEs whose preambles have not been 
        detected will start their backoff periods
        '''
        unidentified_ues = (ue for ue in self.ra_lists[CE_level] if ue.state not in [STATE.RAO, STATE.BACKOFF])
        for ue in unidentified_ues:
            ue.state = STATE.BACKOFF
            ue.ra_attempts += 1
            if ue.ra_attempts > self.preamble_trans_max and ue.CE_level < 2:
                ue.ra_attempts = 0
                ue.CE_level += 1
            backoff_event = Event('Backoff_end', ue = ue)

            self.m.perf_monitor.backoff(CE_level) # performance

            if backoff == 0:
                self.backoff_end(backoff_event)
            else:
                backoff_time = self.rng.integers(backoff)
                if backoff_time == 0:
                    self.backoff_end(backoff_event)
                else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    e_time = t + backoff_time
                    schedule_event(e_time, backoff_event)


    def update_ra_parameters(self, MAC_timer = 64, preamble_trans_max_CE = 2, probability_anchor = 0):
        '''
        method used by NodeB to update the random access parameters of the cell
        '''
        self.preamble_trans_max = preamble_trans_max_CE
        self.m.access_procedure.probability_anchor = probability_anchor
        self.MAC_timer = MAC_timer

    def report_ue_states(self):
        print(' POPULATION RA_list')
        for CE_level in [0,1,2]:
            for ue in self.ra_lists[CE_level]:
                print(f' UE_id={ue.id}: CE_level: {CE_level}, state={UE_states[ue.state]}')
        print('')
        print(' POPULATION Contention list')
        for ue_id, ue in self.contention_ues.items():
            print(f' UE_id={ue_id}: state={UE_states[ue.state]}')
        print('')

    def brief_report(self):
        print(' POPULATION RA')
        for CE_level in [0,1,2]:
            print(f' UEs in CE {CE_level}: {len(self.ra_lists[CE_level])}')
        print('')
        print(' POPULATION Contention list')
        print(f' UE in contention: {len(self.contention_ues)}')
        print('')