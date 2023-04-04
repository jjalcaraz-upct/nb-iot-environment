#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a environment for the OpenAI gym environment

Created on December 13, 2021

@author: juanjosealcaraz
"""

import gym

class Environment(gym.Env):
    ''' 
    class for an environment. It is essentially a wrapper
    '''
    def __init__(self, system = None):
        self.system = system
        self.action_space = system.get_action_space()
        self.observation_space = system.get_obs_space()

    def reset(self):
        """
        Reset the environment 
        """
        state, _ = self.system.reset()

        return state # reward, done, info can't be included

    def step(self, action):
        """
        :action: [int, int, ...] Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional information
        """
        # apply the action
        obs, r, _, info = self.system.step(action)
        return obs, r, False, info

    def render(self):
        pass