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
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    },

    '2250_10_B': {
    'M': 2250,
    'ratio': 1,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    },

    '2500_10_B': {
    'M': 2500,
    'ratio': 1,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    },

    '1500_10_B': {
    'M': 1500,
    'ratio': 1,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    },

    '1750_10_B': {
    'M': 1750,
    'ratio': 1,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    },

    '1000_10_B': {
    'M': 1000,
    'ratio': 1,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    },

    '1250_10_B': {
    'M': 1250,
    'ratio': 1,
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
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    'sort_criterium': 'loss'
    },

    '1000_10_BS': {
    'M': 1000,
    'ratio': 1,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    'sort_criterium': 'loss'
    },

    '3000_10_B': {
    'M': 3000,
    'ratio': 1,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    },

    '3000_10_BS': {
    'M': 3000,
    'ratio': 1,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    'sort_criterium': 'loss'
    },

    '2000_08_B': {
    'M': 2000,
    'ratio': 0.8,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False
    },

    '2000_08_BS': {
    'M': 2000,
    'ratio': 0.8,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False,
    'sort_criterium': 'loss'
    },

    '2000_05_B': {
    'M': 2000,
    'ratio': 0.5,
    'buffer_range': [100, 500],
    'reward_criteria': 'average_delay',
    'statistics': True,
    'animate_carrier': False,
    'animate_stats': False,
    'sc_adjustment': True,
    'mcs_automatic': False
    },

    '2000_05_BS': {
    'M': 2000,
    'ratio': 0.5,
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