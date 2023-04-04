#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class defines different types of wrappers for the environment with the OpenAI gym environment

Created on October, 2022

@author: juanjosealcaraz
"""

import gym
from gym.spaces import Discrete
from itertools import product

def to_discrete(dimensions):
    args = []
    for d in dimensions:
        args.append(range(d))
    return list(product(*args))

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
    
    def reset(self):
        self.obs = self.env.reset()
        return self.obs
        
    def step(self, action):
        n_users = len(self.info['ues'])
        self.n += 1
        if n_users > 1 and action >= n_users: # impossible action
            reward = 10 * self.min_reward
            done = False
        else:
            self.obs, reward, done, self.info = self.env.step(action)
            if reward < self.min_reward:
                self.min_reward = reward # this prevents local minima
        if self.verbose and self.n % self.N == 0:
            self.R += reward
            av_R = self.R / self.n
            time = self.info['time']
            ues = self.info['ues']
            print(f'step {self.n}, time: {time}, average reward: {av_R}, action: {action}, users: {ues}, reward: {reward}')

        return self.obs, reward, done, self.info
    
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
            done = False
        else:
            self.obs, reward, done, self.info = self.env.step(action)
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

        return self.obs, reward, done, self.info


class BasicMcsWrapper(BasicWrapper):
    '''
    basic wrapper for an agent selecting Imcs and Nrep 
    '''
    def __init__(self, env, verbose = False, n_report = 100):
        super().__init__(env, verbose = verbose, n_report = n_report)
        # self.info = {'time': 0, 'receptions': [], 'errors': [], 'unfit': []}
        
    def step(self, action):
        self.n += 1
        obs, _, done, info = self.env.step(action)
        
        # ignore the reward given by the environmet receptions = +1, errors = -1, unfits = -10
        reward = len(info['receptions']) - len(info['errors']) - 10 * len(info['unfit'])
        
        if self.verbose and self.n % self.N == 0:
            self.R += reward
            av_R = self.R / self.n
            time = info['time']
            ues = info['ues']
            print(f'step {self.n}, time: {time}, average reward: {av_R}, action: {action}, users: {ues}, reward: {reward}')

        return obs, reward, done, info

class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.actions = to_discrete(env.action_space.nvec)
        self.action_space = Discrete(len(self.actions))
    
    def action(self, act):
        return list(self.actions[act])
