#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the RxProcedure class

Created on May 6, 2022

@author: juanjosealcaraz

"""

from .event_manager import subscribe

class RxProcedure:
    '''
    Determines if an msg3 or NPUSCH from an UE is received or not.
    It is instantiated with a message switch that gives access to a 
    Population object and a Channel object.
    Communicates with Population useing the NPUSCH_end and msg4 methods.
    Handled events: NPUSCH_end, Msg3.
    '''
    def __init__(self, m):
        self.m = m

        # event subscription
        subscribe('Msg3', self.msg3_event)
        subscribe('NPUSCH_end', self.NPUSCH_end)

    def msg3_event(self, event):
        '''
        processes msg3 messages 
        the event contains contention_list with the contending ues
        '''
        contention_list = event.ue_list
        ue, connected = self.m.channel.msg3_detection(contention_list)
        CE = ue.CE_level

        contenders = len(contention_list)
        
        if connected:
            self.m.node.msg3_outcome(ue, event.t, True) # notify node_b
            # collisions = contenders - 1
            # self.m.perf_monitor.rar_sample(1, 0, collisions, CE, event.t)
            self.m.perf_monitor.rar_sample(1, 0, 0, CE, event.t)
        else:
            self.m.node.msg3_outcome(ue, event.t, False)
            if contenders > 1:
                # self.m.perf_monitor.rar_sample(0, 0, contenders, CE, event.t)
                self.m.perf_monitor.rar_sample(0, 0, 1, CE, event.t)
            else:
                self.m.perf_monitor.rar_sample(0, 1, 0, CE, event.t)
        
    
    def NPUSCH_end(self, event):
        '''
        processes NPUSHC end (data transmited by UEs)
        the event contains the ue
        '''
        ue = event.ue
        SINR, received = self.m.channel.NPUSCH_detection(ue)
        self.m.node.NPUSCH_end(ue.id, event.t, received, SINR) # notify node_b
