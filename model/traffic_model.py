#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions that implement the user arrival model

Created on Dec 5, 2023

@author: juanjosealcaraz

"""
import numpy as np
from numpy.random import default_rng
from scipy.stats import lognorm
from system.channel import Channel, N, F
from system.message_switch import MessageSwitch
from system.user import UE
import matplotlib.pyplot as plt


def signal_samples(n_ues, seed = 345):
    '''
    Generates samples of RSRP signal power, loss and SINR in a cell.
    The user location model in 'channel' considers an hexagonal cell with the 
    antenna placed in one of the vertices
    '''
    sinr_list = np.zeros(n_ues)
    RSRP_list = np.zeros(n_ues)
    loss_list = np.zeros(n_ues)
    rng = default_rng(seed = seed)
    m = MessageSwitch()
    channel = Channel(rng,m)
    for n in range(n_ues):
        ue = UE(n, t_arrival = n, buffer = 256)
        ue = channel.set_xy_loss(ue, range = 10)
        loss_list[n] = ue.loss
        RSRP_list[n] = 35 - ue.loss
        # LogF = rng.normal(0,sigma_F)
        LogF = 0
        rx_pw = ue.tx_pw - ue.loss - LogF # raileigh fading could be added here
        SINR = rx_pw - N - F
        sinr_list[n] = SINR

    return RSRP_list, loss_list, sinr_list


def power_distribution_parameters(RSRP_list, simulate = False):
    '''
    Fits the parameters of a lognornal to the power samples provided 
    '''
    if simulate:
        l_bound, u_bound = -130, -50
        params = lognorm.fit(RSRP_list) 
        x = np.linspace(l_bound,u_bound, 100)
        p = lognorm.pdf(x, *params)

        plt.hist(RSRP_list, bins=50, density=True, range=[l_bound,u_bound])
        plt.xticks(np.arange(l_bound, u_bound, 10))
        plt.plot(x, p, 'k', linewidth=2)
        title = f'params = {params}'
        plt.title(title)
        plt.show()
    else:
        params = (0.7188189705217567, -118.82121523440901, 8.490017897761877)
    return params


def user_partition(th_C1, th_C0, probabilities, th_values):
    '''
    Given the threshold levels for CE0 and CE1, and the parameters of a 
    lognormal distribution for the signal determines the arrival rate 
    for each CE level
    '''
    if th_C0 < th_C1:
        return 0, 0, 0
    
    if th_C0 == th_C1:
        th_C0 = 0 # in this case CE0 level is supressed

    if th_C0 == 0:

        i_ce1 = th_values.index(th_C1)

        P_ce0 = 0

        # Probability of a sample being smaller than th_C1
        P_ce2 = sum(probabilities[:i_ce1+1])

        # Probability of a sample being larger than th_C1
        P_ce1 = 1 - P_ce2


    else:
        i_ce0 = th_values.index(th_C0)

        i_ce1 = th_values.index(th_C1)

        # Probability of a sample being smaller than th_C1
        P_ce2 = sum(probabilities[:i_ce1+1])

        # Probability of a sample being larger than th_C0
        P_ce0 = sum(probabilities[i_ce0+1:])

        # Probability of a sample being larger than th_C1 and smaller than th_C0
        P_ce1 = 1 - P_ce2 - P_ce0   

    return P_ce0, P_ce1, P_ce2


def arrival_flows_estimation(th_C1, th_C0, arrival_rate, probabilities, th_values):
    '''
    determines the incoming flow to NPRACH channels per CE level
    '''
    t_fractions = user_partition(th_C1, th_C0, probabilities, th_values) # traffic fractions

    arrival_rate_per_CE = [arrival_rate * fraction for fraction in t_fractions] # arrival_ rates

    return arrival_rate_per_CE