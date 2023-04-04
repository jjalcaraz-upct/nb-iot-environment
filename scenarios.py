#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of simulated scenarios.
Esach scenario is identified by a name.

Created on Nov 11, 2022

@author: juanjosealcaraz

"""

scenarios = {
    '2000_10_B': {
    'M': 2000,
    'ratio': 1,
    'levels': [0,1,2],
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    },

    '2000_10_BS': {
    'M': 2000,
    'ratio': 1,
    'levels': [0,1,2],
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    'sort_criterium': 'loss'
    }
}