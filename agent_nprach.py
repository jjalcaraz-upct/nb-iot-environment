#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Model-based agent

@author: juanjosealcaraz

"""
# import time
# import numpy as np
# from itertools import product
# import matplotlib.pyplot as plt
import system.parameters as par
from control_agents import DummyAgent
from model.detection_model import estimate_incoming_traffic
import model.random_access_model as ra
import model.configurator as con
import math
import json

min_value = 0.0035
max_value = 0.374
step = 0.002

with open('rate_to_conf.json', 'r') as file:
    rate_to_conf = json.load(file)

# threshold-dependent parameters
th_C0_i = par.control_default_values['th_C0']
th_C1_i = par.control_default_values['th_C1']
th_C0 = par.th_values[th_C0_i]
th_C1 = par.th_values[th_C1_i]
msg3_RUs = ra.compute_msg3_rus(th_C1, th_C0)
N_sfs_list = ra.compute_NPRACH_sf(th_C1, th_C0)    

def get_surrounding_elements(lst, index):
    # Using list slicing to handle edge cases
    return lst[max(0, index - 1):min(index + 2, len(lst))]

class NPRACH_THAgent(DummyAgent):
    '''
    agent that controls NPRACH parameters by exploiting a system model
    '''
    def __init__(self, dict, metrics, verbose = False, buffer_size = 100, margin = 1.8, no_th = False):
        super().__init__(dict)
        self.verbose = verbose
        self.samples = {metric: [] for metric in metrics}
        self.n = 0
        self.buffer_size = buffer_size
        self.k = 0
        self.no_th = no_th
        self.margin = margin
        self.lambda_estimate = 0.01
        self.msg3_detection = [0.9,0.9,0.9]
        self.conf = [2,2,2,2,2,2] # intial configuration: nsc0, nsc1, nsc2, per0, per1, per2
        self.th_conf = [th_C1_i, th_C0_i] # initial configuration for th_C1 and th_C0
    
    def set_parameter(self, margin):
        self.margin = margin

    def get_action(self, obs, r, info, action):
        '''
        the action contains ['th_C1', 'th_C0', 'sc_C0', 'sc_C1', 'sc_C2', 'period_C0', 'period_C1', 'period_C2']
        '''
        # estimate incoming traffic
        detections = info['NPRACH_detection']
        collisions = info['NPRACH_collision']
        loss_samples = info['incoming']
        # connected_ues = info['total_ues']
        # msg3_detect_ratios = info['msg3_detection']
        self.n += 1
        ce_arrival_rates = [0.0, 0.0, 0.0]
        self.lambda_estimate = 0.0
        for ce in range(3):
            N_sc = par.N_sc_list[self.conf[ce]]
            period = par.period_list[self.conf[ce + 3]]
            d_ = detections[ce]
            c_ = collisions[ce]
            a_ = estimate_incoming_traffic(d_, c_, N_sc)
            lambda_i = a_ / period
            self.lambda_estimate += lambda_i
            # APPLY THE MARGIN HERE !
            ce_arrival_rates[ce] = self.margin * lambda_i # in arrivals per sf (ms)
            #
            # msg3_detection[ce] += (msg3_detect_ratios[ce] - msg3_detection[ce])/self.n
        
        arrival_rate = sum(ce_arrival_rates) # estimated with a margin

        # update probability distribution
        if self.k < self.buffer_size: # still gathering samples to build the model
            self.k += len(loss_samples)
            distribution = info['distribution']
            self.msg3_detection = ra.msg3_detection_estimation(th_C1, th_C0, distribution)
            con.update_parameters(msg3_RUs, N_sfs_list, self.msg3_detection)
            conf = con.configurator(self.msg3_detection, ce_arrival_rates)
            th_conf = self.th_conf
        
        elif self.no_th: # the agent controls only the NPRACH parameters
            conf = con.configurator(self.msg3_detection, ce_arrival_rates)
            th_conf = self.th_conf
        
        else: # the agent controls the CE thresholds and NPRACH parameters
            if arrival_rate >= min_value and arrival_rate <= max_value:
                index = min(int((arrival_rate - min_value)//step) + 1, len(rate_to_conf)-1 )
                [th, conf] = rate_to_conf[index]
                th_conf = par.th_indexes[(th[0], th[1])]
            elif arrival_rate < min_value:
                [th, conf] = rate_to_conf[0]
                th_conf = par.th_indexes[(th[0], th[1])]
            else:
                [th, conf] = rate_to_conf[-1]
                th_conf = par.th_indexes[(th[0], th[1])]
                self.conf = conf
                self.th_conf = th_conf
       
        # store samples from previous step
        for metric, samples in self.samples.items():
            sample = info.get(metric, self.margin) # if the metric is not in info, the metric is the margin (beta) factor
            if isinstance(sample, list):
                # Avoid division by zero for empty lists
                avg = sum(sample) / len(sample) if sample else 0
                samples.append(avg)
            else:
                # If the value is not a list, store it as is
                samples.append(sample)
        
        # output
        if self.verbose:
            [th_1_i, th_0_i] = th_conf
            th_CE1 = par.th_values[th_1_i]
            th_CE0 = par.th_values[th_0_i]
            print('---------------------------------------------------------')
            print(f'step: {self.n}')
            for k,v in info.items():
                if k in ['time', 'incoming', 'total_ues', 'delays', 'departures', 'NPRACH_detection', 'NPRACH_collision']:
                    print(f' {k}: {v}')
            print(f' AGENT estimates arrival rates: {ce_arrival_rates} -> {arrival_rate}')
            print(f'  |_ reward: {r}')
            print(f'  |_ loss samples: {self.k}')
            print(f'  |_ margin: {self.margin}')
            print(f'  |_ selects th: {th_CE1, th_CE0}')
            print(f'  |_ selects action: {conf}')
            print('---------------------------------------------------------')
        # return list(self.conf)
        return th_conf + conf
    
    
class NPRACH_Bandit_Agent(NPRACH_THAgent):
    def __init__(self, dict, metrics, verbose = False, margin = 1.8, box = [1.4, 3.4], step = 0.2, alpha = None, no_th = False):
        super().__init__(dict, metrics, verbose = verbose, margin = margin, no_th = no_th)
        self.margins = [box[0] + i * step for i in range(int((box[-1] - box[0]) / step + 1))]
        self.n_arms = len(self.margins) 
        self.counts = [0] * self.n_arms
        self.values = [0] * self.n_arms
        self.alpha = alpha
        self.total_count = 0
        self.chosen_arm = 0

    def select_arm(self):
        # If any arm hasn't been played, play it
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self._calculate_ucb_values()
        return ucb_values.index(max(ucb_values))

    def update(self, reward):
        # Update the counts and values with the new reward
        chosen_arm = self.chosen_arm
        self.counts[chosen_arm] += 1
        self.total_count += 1
        n = self.counts[chosen_arm]
        step_size = 1/n
        if self.alpha:
            step_size = max(self.alpha, step_size)
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) * step_size

    def _calculate_ucb_values(self):
        return [self.values[arm] + math.sqrt((2 * math.log(self.total_count)) / self.counts[arm]) for arm in range(self.n_arms)]

    def get_action(self, obs, r, info, action):
        self.update(r)
        self.chosen_arm = self.select_arm()
        self.margin = self.margins[self.chosen_arm]
        return super().get_action(obs, r, info, action)


class NPRACH_UBandit_Agent(NPRACH_Bandit_Agent):
    '''
    unimodal version of the bandit agent
    '''
    def __init__(self, dict, metrics, verbose = False, box = [1.4, 3.4], step = 0.2, alpha = None):
        super().__init__(dict, metrics, verbose = verbose, box = box, step = step, alpha = alpha)
    
    def select_arm(self):
        # If any arm hasn't been played, play it
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # find the leader
        leader = self.values.index(max(self.values))

        # get the ucb values
        ucb_values = self._calculate_ucb_values()
        ucb_u_values = get_surrounding_elements(ucb_values, leader)

        return ucb_values.index(max(ucb_u_values))

