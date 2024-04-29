#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the UE as a dataclass

Created on Jan 13, 2022

@author: juanjosealcaraz

"""

from dataclasses import dataclass
from collections import namedtuple
# from typing import List

def initiate_list():
    return [0, 0, 0]

UE_states = ['RAO', 'BACKOFF', 'RA', 'RAR', 'CAPTURE', 'COLLIDED', 'CONREQ', 'CONNECTED', 'DATA', 'DONE']

state_numbers = list(range(len(UE_states)))

state_dict = dict(zip(UE_states, state_numbers))

State = namedtuple('State', UE_states)

# UE States
STATE = State(**state_dict)

@dataclass
class UE:
    '''
    Stores UE information.
    '''
    id: int

    state: int = 0

    # access variables
    beta: bool = False
    CE_level: int = 0 
    t_arrival: int = 0 # arrival time
    ra_attempts: int = 0 # random access attempts in this CE_level
    preamble: int = 0 # selected ra preamble
    rar_w_id: int = 0 # identifier of the RAR window
    # prob_anchor: float = 0.0 # probability of selecting non-anchor carrier
    
    # performance variables
    t_contention: int = 0 # reference time for the contention resolution (msg3 grant received)
    t_connection: int = 0 # when the UE entered connection mode (msg4 received)
    t_disconnection: int = 0 # when the UE disconnected
    backoff_counter: int = 0
    timeout_counter: int = 0
    harq_rtx: int = 0

    # data variables
    buffer: int = 256
    retx_buffer: int = 0
    x: float = 0.0
    y: float = 0.0
    sinr: float = 0.0 # for HARQ
    tx_pw: float = 23 # 23 dBm = 200 mW
    loss: float = 70 # dB
    I_tbs: int = 0
    N_rep: int = 0
    N_ru: int = 0
    bits: int = 0
    tbs: int = 0 # transport block size
    new_data: bool = True
    acked_data: int = 0

if __name__ == '__main__':
    from random import choice
    ue_list = []
    for id in range(10):
        ue_list.append(UE(id, CE_level = choice([0,1,2])))
    
    print([ue for ue in ue_list if ue.CE_level == 0])
    print([ue for ue in ue_list if ue.CE_level == 1])
    print([ue for ue in ue_list if ue.CE_level == 2])
