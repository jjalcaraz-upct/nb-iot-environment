#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class generates a environment for the OpenAI gym environment

Created on December 13, 2021

@author: juanjosealcaraz
"""

import gymnasium as gym

class Environment(gym.Env):
    ''' 
    class for an environment. It is essentially a wrapper
    '''
    def __init__(self, render_mode=None, system = None):
        self.system = system
        self.action_space = system.get_action_space()
        self.observation_space = system.get_obs_space()

    def reset(self, seed=None, options=None):
        """
        Reset the environment 
        """
        state, info = self.system.reset()

        return state, info

    def step(self, action):
        """
        :action: [int, int, ...] Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional information
        """
        # apply the action
        obs, r, _, _, info = self.system.step(action)
        return obs, r, False, False, info

    def render(self):
        pass