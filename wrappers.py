#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class defines different types of wrappers for the environment with the OpenAI gym environment

Created on October, 2022

@author: juanjosealcaraz
"""

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete, Box
import system.parameters as par
from system.carrier import compute_offsets
from itertools import product
from system.utils import read_flag_from_file
from collections import deque
import numpy as np
import time

first_part = np.linspace(0.0035, 0.1, 20, endpoint=False)  # Exclude the endpoint to avoid repetition
second_part = np.linspace(0.1, 0.2, 10)

s_action_values = np.concatenate((first_part, second_part))

def to_discrete(dimensions):
    args = [range(d) for d in dimensions]  # List comprehension to generate ranges
    return [list(item) for item in product(*args)]  # Convert each tuple to a list

def convert_seconds(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return hours, minutes, seconds

def append_to_file(filename, line):
    with open(filename, "a") as file:
        file.write(line)

# default values
th_C0_i = par.control_default_values['th_C0']
th_C1_i = par.control_default_values['th_C1']
th_C0 = par.threshold_list[th_C0_i]
th_C1 = par.threshold_list[th_C1_i]
N_sfs_list = par.compute_NPRACH_sf(th_C1, th_C0)

class BasicWrapper(gym.Wrapper):
    '''
    basic wrapper for an agent selecting UEs 
    The wrapper's objective is to avoid selecting unexisting ues
    '''
    def __init__(self, env, verbose = False, n_report = 100):
        super().__init__(env)
        self.env = env
        self.n = 0
        self.R = 0
        self.verbose = verbose
        self.N = n_report
        self.min_reward = -100
        self.info = {'time': 0, 'ues':[0]}
    
    def reset(self, seed = None, options = None):
        self.obs, info = self.env.reset()
        return self.obs, info
        
    def step(self, action):
        n_users = len(self.info['ues'])
        self.n += 1
        if n_users > 1 and action >= n_users: # impossible action
            reward = 10 * self.min_reward
            terminated = False
            truncated = False
        else:
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            if reward < self.min_reward:
                self.min_reward = reward # this prevents local minima
        if self.verbose and self.n % self.N == 0:
            self.R += reward
            av_R = self.R / self.n
            time = self.info['time']
            ues = self.info['ues']
            print(f'step {self.n}, time: {time}, average reward: {av_R}, action: {action}, users: {ues}, reward: {reward}')

        return self.obs, reward, terminated, truncated, self.info
    
class BasicSchedulerWrapper(BasicWrapper):
    '''
    basic wrapper for an agent selecting the UE, and its Imcs, Nrep 
    '''
    def __init__(self, env, verbose = False, n_report = 100):
        super().__init__(env, verbose = verbose, n_report = n_report)
        self.info = {'time': 0, 'ues':[0], 'unfit': []}
        
    def step(self, action):
        n_users = len(self.info['ues'])
        self.n += 1
        if n_users > 1 and action[0] >= n_users: # impossible action
            reward = 10 * self.min_reward # perhaps could be removed
            terminated = False
            truncated = False
        else:
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            if reward < self.min_reward:
                self.min_reward = reward # this prevents local minima
        
        if len(self.info['unfit']) > 0: # penalty for selecting Imcs and Nrep that does not fit
            reward += self.min_reward
        
        if self.verbose and self.n % self.N == 0:
            self.R += reward
            av_R = self.R / self.n
            time = self.info['time']
            ues = self.info['ues']
            print(f'step {self.n}, time: {time}, average reward: {av_R}, action: {action}, users: {ues}, reward: {reward}')

        return self.obs, reward, terminated, truncated,  self.info


class NPRACH_wrapper(gym.Wrapper):
    '''
    Wrapper for an agent selecting CE Thresholds and NPRACH parameters
    '''
    def __init__(self, env, discrete = False, verbose = False, use_default_action = False, n_report = 100, norm = 100):
        super().__init__(env)
        self.env = env
        self.verbose = verbose
        self.default_action = use_default_action
        self.norm = norm
        self.N = n_report
        self.n = 0
        self.action = [5,8,2,2,2,2,2,2]
        self.info = {'time': 0,
                     'state': 'NPRACH_update', 
                     'total_ues': 0,
                     'preambles': [12, 12, 12],
                     'NPRACH_detection': [0, 0, 0],
                     'NPRACH_collision': [0, 0, 0],
                     'msg3_detection': [0, 0, 0],
                     'msg3_sent': [0, 0, 0],
                     'msg3_received': [0, 0, 0],
                     'RA_occupation': 0,
                     'NPUSCH_occupation': 0,
                     'NPRACH_occupation': 0,
                     'incoming': [],
                     'delays': [],
                     'av_delay': 0,
                     'departures': 0,
                     'service_times': [],
                     'NPRACH conf': []
                    }
        # modify action space to include only feasible threhold pairs
        max_actions = env.action_space.nvec[1:]
        max_actions[0] = len(par.th_pairs)
        
        self.discrete = discrete        
        if discrete:
            self.actions = to_discrete(max_actions)
            self.action_space = Discrete(len(self.actions))
        else:
            self.action_space = MultiDiscrete(max_actions)
    
    def reset(self, seed = None, options = None):
        self.obs, info = self.env.reset()
        return self.obs, info
        
    def step(self, action):
        self.n += 1 # step counter
        if self.discrete:
            action = list(self.actions[action])

        #extract th pair
        th_pair_i = action[0]
        (th_C1, th_C0) = par.th_pairs[th_pair_i]

        # extract NPRACH configuration
        nsc = [par.N_sc_list[i] for i in action[1:4]]
        periods = [par.period_list[i] for i in action[4:]]

        # check if valid configuration
        N_sfs_list = par.compute_NPRACH_sf(th_C1, th_C0)
        _, _, _, valid = compute_offsets(periods, N_sfs_list, nsc)

        if not valid:
            # simply apply default action (previous valid action)
            action = self.action
            reward = -1
            self.obs, _, terminated, truncated, self.info = self.env.step(action)
        else:
            # transform action
            action = par.th_indexes[(th_C1, th_C0)] + list(action[1:])
            # apply action
            self.obs, reward, terminated, truncated, self.info = self.env.step(action)
            reward = reward / self.norm

        if not self.default_action:
            self.action = action # update default action
        
        if self.verbose and self.n % self.N == 0:
            time = self.info['time']
            if valid:
                print(f' t: {time} reward: {reward} AGENT selected action {action}')
            if not valid:
                print(f' t: {time} reward: {reward} AGENT selected NON VALID action {action}')
            for k,v in self.info.items():
                print(f'     {k}: {v}')
            print('-------------------------------------------------------------------------------------------')
            
        return self.obs, reward, terminated, truncated, self.info   

class NPRACH_wrapper_traces(NPRACH_wrapper):
    def __init__(self, env, metrics, discrete = False, verbose = False, experiment = 0, logfile = None, flagfile = None, n_report = 100, norm = 100):
        '''
        metric is a list with the metrics we want to store
        '''
        self.logfile = logfile
        self.flagfile = flagfile
        self.instance_id = experiment
        self.samples = {metric: [] for metric in metrics}
        self.averages = {metric: 0 for metric in metrics}
        if logfile:
            self.start_time = time.time()
        super().__init__(env, verbose = verbose, discrete = discrete, n_report = n_report, norm = norm)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        for metric, samples in self.samples.items():
            sample = info.get(metric, None)
            if isinstance(sample, list):
                # Avoid division by zero for empty lists
                sample = sum(sample) / len(sample) if sample else 0
                samples.append(sample)
            else:
                # If the value is not a list, store it as is
                samples.append(sample)
            if self.logfile:
                self.averages[metric] += (sample - self.averages[metric])/self.n # incremental estimation
        
        # store in a logfile
        check_point = self.n % self.N == 0
        if self.instance_id == 0 and self.logfile and check_point:
            elapsed_time = time.time() - self.start_time
            h, m, s = convert_seconds(elapsed_time)
            line = f'{self.n}:'
            for metric, avg in self.averages.items():
                line += f' {metric}: {avg:.2f} |'
            line += f'| {int(h)}:{int(m)}:{int(s)}\n'
            append_to_file(self.logfile, line)
        
        # check continuation
        if self.flagfile and check_point:
            terminated = read_flag_from_file(self.flagfile)

        return obs, reward, terminated, truncated, info

obs_items = [11, 12] # total_ues, av_delay

class NPRACH_agent_wrapper(gym.Wrapper):
    '''
    Wrapper for an RL agent that configures only the margin (beta) value 
    of the model-based configurator
    '''
    def __init__(self, env, agent, bounds = [1.4, 3.4], reduced_observation = True, discrete = True, n_actions = 30, obs_items = obs_items, obs_window = 1, norm = 100):
        super().__init__(env)
        if discrete:
            self.action_space = Discrete(n_actions)
            self.action_values = np.linspace(bounds[0], bounds[1], n_actions)
        else:
            self.action_space = Box(low = bounds[0], high = bounds[1])
        self.discrete = discrete
        if reduced_observation:
            obs_size = obs_window + len(obs_items)
            self.obs_items = obs_items
            self.observation_space = Box(0, 1, shape=(obs_size,))
            self.arrival_history = deque([0] * obs_window, maxlen = obs_window)
        self.reduced_observation = reduced_observation
        self.agent = agent
        self.samples = agent.samples
        self.r = 0
        self.norm = norm
        self.conf = [0, 0, 0, 0, 0, 0]
        self.n = 0

    def reset(self, seed = None, options = None):
        obs, self.info = self.env.reset()
        if self.reduced_observation:
            obs = [obs[i] for i in self.obs_items] # RA occupation & total_ues
            obs.extend(self.arrival_history) # arrivals
        self.obs = obs
        return self.obs, self.info

    def step(self, action):
        self.n += 1 # step counter
        
        # get action
        if self.discrete:
            action = self.action_values[action]
        else:
            action = action[0]
        
        # apply action                 
        self.agent.set_parameter(action)

        # get the configuration
        conf = self.agent.get_action(self.obs, self.r, self.info, self.conf)

        # apply the action to the environment
        obs, reward, terminated, truncated, self.info = self.env.step(conf)
        self.conf = conf
        self.r = reward / self.norm
        
        if self.reduced_observation:
            obs = [obs[i] for i in self.obs_items] # RA occupation / total_ues
            self.arrival_history.append(self.agent.lambda_estimate) # current arrival estimation
            obs.extend(self.arrival_history)  # arrivals

        self.obs = obs

        return self.obs, self.r, terminated, truncated, self.info
    

class NPRACH_agent_wrapper_traces(NPRACH_agent_wrapper):
    def __init__(self, env, agent, bounds = [1.4, 3.4], experiment = 0, logfile = None, n_report = 100, reduced_observation = True, discrete = True, n_actions = 30, obs_window = 1, norm = 100):
        '''
        metric is a list with the metrics we want to store
        '''
        self.instance_id = experiment
        self.logfile = logfile
        self.N = n_report
        super().__init__(env, agent, bounds = bounds, reduced_observation = reduced_observation, discrete = discrete, n_actions = n_actions, obs_window = obs_window, norm = norm)
        self.averages = {metric: 0 for metric in self.samples.keys()}
        if logfile:
            self.start_time = time.time()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.logfile:
            for metric, samples in self.samples.items():
                sample = samples[-1] # pick the last sample
                self.averages[metric] += (sample - self.averages[metric])/self.n # incremental estimation
        
        # store in a logfile
        check_point = self.n % self.N == 0
        if self.instance_id == 0 and self.logfile and check_point:
            elapsed_time = time.time() - self.start_time
            h, m, s = convert_seconds(elapsed_time)
            line = f'{self.n}:'
            for metric, avg in self.averages.items():
                line += f' {metric}: {avg:.2f} |'
            line += f'| {int(h)}:{int(m)}:{int(s)}\n'
            append_to_file(self.logfile, line)

        return obs, reward, terminated, truncated, info


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.actions = to_discrete(env.action_space.nvec)
        self.action_space = Discrete(len(self.actions))
    
    def action(self, act):
        return list(self.actions[act])


class RAR_wrapper(gym.Wrapper):
    def __init__(self, env, reward_criteria = 'msg3'):
        super().__init__(env)
        self.env = env
        self.reward_criteria = reward_criteria
        self.reward_history = []
        self.detection_history = []
        self.departure_history = []
        self.total_msg_t_1 = 0
        self.total_departures_1 = 0
        self.ce_level_selected = []
        self.I_mcs_history = [[], [], []]
        self.N_rep_history = [[], [], []]
        self.n_steps = 0
    
    def reset(self, seed = None, options = None):
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        t = info['time']
        RAR_detected = info['RAR_detected'] # Counters of msg3s detected per CE level
        total_msg_t = sum(RAR_detected) # Counter of all msg3 detected up to t
        detections = max(0, total_msg_t - self.total_msg_t_1) # Detections between t and (t-1)
        self.total_msg_t_1 = total_msg_t

        RAR_failed = info['RAR_failed']
        total_failed = sum(RAR_failed) # msg3 not delivered on time

        total_departures = info['total_departures']
        departures = max(0, total_departures - self.total_departures_1)
        self.total_departures_1 = total_departures 

        # reward estimation
        if not info['valid_CE_control']:
            reward = - 1
        elif self.reward_criteria == 'departures': # departure-based reward
            reward = max(-1, min(1, departures/4) - total_failed/4)           
        else: # connection-based reward
            reward = max(-1, min(1, detections/4) - total_failed/4)

        # sampling actions
        action = info['action']
        CE_level = action[0]
        I_mcs = action[1]
        N_rep = action[2]
        self.ce_level_selected.append(CE_level)
        self.I_mcs_history[CE_level].append(I_mcs)
        self.N_rep_history[CE_level].append(N_rep)
        self.reward_history.append(reward)
        self.departure_history.append((t, departures))
        self.detection_history.append((t, detections))

        return obs, reward, terminated, truncated, info