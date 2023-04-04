#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements a block fading model, where a channel realization is constant 
over a NPUSCH transmission, and changes independently from one transmission to another
The bler error rate of a NPUSCH transmission is estimated from SINR, I_tbs, N_rep
using the tables obtained with "Lite NBIoT NPUSCH Simulator"
https://github.com/CSC-CONICET/Lite-NBIoT-NPUSCH-Simulator 

Created on Feb 20, 2022

@author: juanjosealcaraz

"""

import numpy as np
import pandas as pd
from itertools import product
from parameters import Imcs_to_Itbs, N_rep_list
from user import STATE

''' Based on Hwang, Jeng-Kuang, Cheng-Feng Li, and Chingwo Ma. 
"Efficient detection and synchronization of superimposed NB-IoT NPRACH preambles." 
IEEE Internet of Things Journal 6.1 (2018): 1173-1182.'''
N = -121.4 # Noise in dBm
Tx_pw = 23 # dBm
Gmax = 15 #dBi
sigma_F = 10 # dB standard deviation of the log Normal fading
maxRange = 5 # Km (maximum cell range)
L_o = 0 # penetration loss (walls, ground, etc)
F = 3 # Noise Figure in dB
f = 10**(F/10) # noise figure linear
n_ = f*10**(N/10) # Noise in mW
radius = 1/2

# parameters for NPRACH detection probability curves

nprach_paramerters = {
    1: [-26.65585077,   0.32267874],
    2: [-28.25585118,   0.32267874],
    4: [-29.85584994,   0.32267875],
    8: [-31.45585117,   0.32267874],
    16: [-33.05585139,   0.32267859],
    32: [-34.65584975,   0.32267876],
    64: [-36.25585128,   0.3226787 ],
    128: [-37.85585128,   0.3226787],
}

def sigmoid(x, x0 = 0, k = 1):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return (y)

#  NPUSCH BLER curves

df = pd.read_csv('LUT2.csv')

bler_curves = {}

for itbs,nr in product(Imcs_to_Itbs, N_rep_list):
    snr = df[(df['itbs'] == itbs) & (df['nr'] == nr )][['snr']].to_numpy()
    bler = df[(df['itbs'] == itbs) & (df['nr'] == nr )][['bler']].to_numpy()
    bler_curves[(itbs, nr)] = {round(s[0],1):b[0] for s,b in zip(snr,bler)}

def get_bler(snr, itbs, nr):
    snr = min(max(snr, -30),20) # [-30, 20]
    s = round(snr, 1)
    return bler_curves[(itbs, nr)][s]

''' auxiliary functions defining a hexagonal cell '''
def find_y_value(x1, y1, x2, y2, x):   
    m = (y2 - y1) / (x2 - x1)  # slope
    b = -m * x1 + y1 # y-intercept
    y = m * x + b # y-value
    return y

def lower_left(x):
    return find_y_value(0, 0.5, 0.25, 0, x)

def lower_right(x):
    return find_y_value(0.75, 0, 1, 0.5, x)

def upper_left(x):
    return find_y_value(0, 0.5, 0.25, 1, x)

def upper_right(x):
    return find_y_value(0.75, 1, 1, .5, x)

def location(x, y):
    x_t = x - radius / 2
    distance = np.sqrt(x_t**2 + y**2)
    cos_theta = x_t / distance
    theta = np.arccos(cos_theta)
    theta = np.degrees(theta) - 60
    return distance, theta

def generate_xy(rng):
    in_cell = False
    x, y = 0.1, 0.1
    while not in_cell:
        [x, y] = rng.random(2) 
        in_cell = (y > lower_left(x)) and (y > lower_right(x)) and (y < upper_left(x)) and (y < upper_right(x))
    return x, y

def antenna_pattern(theta):
    ''' provides the gain of the antenna according to TS 36.942 section 4.2 Antenna models'''
    return -1*min(12*(theta/65)**2, 20)
    
class Channel:
    '''
    Class that holds all the required methods for channel simulation
    '''
    def __init__(self,rng, m):
        self.rng = rng
        self.m = m
        self.m.set_channel(self)

    def set_xy_loss(self, ue, A = 128.1, B = 37.6): # A = 120.9 for 900 MHz
        # TS 36.942 section 4.5 Propagation conditions and channel models (2000 MHz)
        x, y = generate_xy(self.rng)
        d, theta = location(x, y)
        R = max(d*maxRange, 0.1)
        G = Gmax + antenna_pattern(theta)
        L = A + B * np.log10(R)
        # gamma = 2.6
        # Free Space Path Loss (R in Km, f in GHz)
        # FSPL = 20*np.log10(2) + 92.45 + gamma*10*np.log10(R) 
        FSPL = 0
        L = max(L, FSPL) - G + L_o
        ue.loss = L
        ue.tx_pw = Tx_pw
        ue.x = x
        ue.y = y

    def preamble_detection(self, ues, N_rep):
        if len(ues) > 1:
            return self.multiple_preamble_detection(ues, N_rep)
        ue = ues[0]
        LogF = self.rng.normal(0,sigma_F) # shadow fading for all the repetitions
        rx_pw = ue.tx_pw - ue.loss - LogF # raileigh fading could be added here
        SINR = rx_pw - N - F
        par = nprach_paramerters[N_rep]
        p = sigmoid(SINR, *par) # detection probability
        detected = self.rng.binomial(1, p)
        if detected:
            ue.state = STATE.RAR
        return detected

    def sinr_estimation(self, ues):
        '''
        returns the max SINR when multiple signals collide
        '''
        pw = [] # received powers
        max_rx = -np.inf
        max_i = 0
        for i, ue in enumerate(ues):
            LogF = self.rng.normal(0,sigma_F) # shadow fading for all the repetitions
            rx_pw = ue.tx_pw - ue.loss - LogF # raileigh fading could be added here
            rx_pw = 10**(rx_pw/10) # power in mW
            pw.append(rx_pw) # power in mW
            if rx_pw > max_rx:
                max_rx = rx_pw
                max_i = i
        pw.remove(max_rx)

        ni = n_ + sum(pw) # noise plus interference
        sinr = max_rx / ni # signal to interference ratio
        SINR = 10*np.log10(sinr) # in dB
        return SINR, max_i

    def multiple_preamble_detection(self, ues, N_rep):
        SINR, _ = self.sinr_estimation(ues)

        par = nprach_paramerters[N_rep]
        p = sigmoid(SINR, *par) # detection probability
        detected = self.rng.binomial(1, p)

        state = STATE.COLLIDED
        if detected:
            state = STATE.CAPTURE
        
        for ue in ues:
            ue.state = state

        return detected

    def msg3_detection(self, contention_list):
        '''
        method used by rx_procedure to check if a msg3 has been detected
        '''
        SINR, i = self.sinr_estimation(contention_list)

        I_tbs = contention_list[0].I_tbs
        N_rep = contention_list[0].N_rep
        p = 1.0 - get_bler(SINR, I_tbs, N_rep)
        detected = self.rng.binomial(1, p)

        return contention_list[i], detected

    def NPUSCH_detection(self, ue):
        '''
        method used by rx_procedure to check if a NPUSCH transmission has been detected
        '''
        I_tbs = ue.I_tbs
        N_rep = ue.N_rep
        tbs = ue.tbs # determined by the N_ru 
        LogF = self.rng.normal(0,sigma_F) # shadow fading for all the repetitions
        rx_pw = ue.tx_pw - ue.loss - LogF # raileigh fading could be added here
        SINR = rx_pw - N - F
        if not ue.new_data: # HARQ to the rescue
            old_sinr = ue.sinr
            SINR = 10*np.log10(10**(SINR/10) + 10**(old_sinr/10))
        p = 1.0 - get_bler(SINR, I_tbs, N_rep)
        if tbs != 256:
            bler_256 = get_bler(SINR, I_tbs, N_rep)
            p = (1.0 - bler_256)**(tbs/256)
        detected = self.rng.binomial(1, p)
        return SINR, detected
    
if __name__=='__main__':
    from parameters import CE_thresholds, SRP
    from numpy.random import default_rng
    from message_switch import MessageSwitch
    from ue_generator import UEGenerator
    from user import UE
    rng = default_rng()
    m = MessageSwitch()
    generator = UEGenerator(rng, m, M = 2000)
    channel = Channel(rng,m)
    CE0_count = 0
    CE1_count = 0
    CE2_count = 0
    ue_count = 0
    for t in range(10000):
        ue_count += 1
        ue = UE(ue_count, t_arrival = t, buffer = 256)
        channel.set_xy_loss(ue)
        P_RSRP = SRP - ue.loss
        if P_RSRP < CE_thresholds[0]:
            CE_level = 2
            CE2_count += 1
        elif P_RSRP < CE_thresholds[1]:
            CE_level = 1
            CE1_count += 1
        else:
            CE_level = 0
            CE0_count += 1
        ue.CE_level = CE_level

    print(f'total ue: {ue_count}')
    print(f'CE0: {100*CE0_count/ue_count}, CE1: {100*CE1_count/ue_count}, CE2: {100*CE2_count/ue_count}')
    total = 6 + 36 + 24
    print(f'total {total}')
    print(f'CE0: {100*6/total}, CE1: {100*36/total}, CE2: {100*24/total}')