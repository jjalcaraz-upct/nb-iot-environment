#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for simulation and presentation of results.

Created on May 2022

@author: juanjosealcaraz

"""

import numpy as np
from scipy import stats
from . import parameters as par
# from .parameters import control_items, control_max_values, control_default_values, state_dim, N_carriers, Horizon

carrier_i = par.control_items['carrier']
id_i = par.control_items['id']
sc_i = par.control_items['sc']

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
