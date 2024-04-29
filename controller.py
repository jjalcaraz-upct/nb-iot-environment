#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the controller class that orchestrates the operation of multiple control agents

Created on June 2022

@author: juanjosealcaraz

"""

import numpy as np
from gymnasium import spaces
import system.parameters as par
from system.utils import read_flag_from_file
from control_agents import DummyAgent, agents_conf

# DEBUG
DEBUG = False

default_agents = [DummyAgent(conf) for conf in agents_conf]

class Controller:
    '''
    This class orchestrates all the agents that concurrently control the system.
    Inputs: a node to control, a list of (internal) agents and an indicator of external agent.
    Each internal agent specifies its own configuration.
    It implements the openai gym interface with the external agent.
    '''
    def __init__(self, node, agents = default_agents, ext_agent = -1):
        self.node = node
        self.agents = agents
        self.action = np.array([0]*par.N_actions, dtype=int)
        self.obs = np.array([0], dtype=float) # dummy
        self.soft_reset = False
        self.reward = 0.0
        self.info = {}
        self.set_ext_agent(ext_agent)
        self._set_agents_ids()
        self._set_obs_dimensions()
        self._initialize_action()

    def _set_agents_ids(self):
        '''
        determine which agents act in each state
        '''
        self.agents_ids = {}
        for agent in self.agents:
            states = agent.states
            for state in states:
                if state in self.agents_ids.keys():
                    self.agents_ids[state].append(agent.id)
                else:
                    self.agents_ids[state] = [agent.id]
    
    def _set_obs_dimensions(self):
        '''
        stores the dimension of the observation for each agent
        '''
        self.obs_dimension = {}
        for agent in self.agents:
            self.obs_dimension[agent.id] = self.get_obs_dimension(agent.obs_items)
    
    def _initialize_action(self):
        '''
        initial action
        '''
        for a in self.agents: # a state may have more than one agent
            self.update_action(a.id)

    def replace_agent(self, new_agent, pos):
        '''
        to insert a trained agent
        '''
        self.agents[pos] = new_agent

    def get_action_space(self):
        '''
        for open ai gym compatibility
        '''
        a_max = self.ext_agent.a_max
        if len(a_max) > 1:
            space = spaces.MultiDiscrete(np.array(a_max) + 1)
        else:
            space = spaces.Discrete(a_max[0] + 1)
        return space
    
    def get_obs_dimension(self, obs_items):
        '''
        gets the dimension of the observation space
        '''
        dim = 0
        for item in obs_items:
            dim += par.obs_dict[item][0]
        return dim

    def get_obs_space(self):
        '''
        for open ai gym compatibility
        '''
        dimension = self.obs_dimension[self.ext_agent.id]
        space = spaces.Box(low=0.0, high=1.0, shape=(dimension,), dtype=float)
        return space

    def get_obs(self, obs_items):
        '''
        extracts items from self.info and constructs the observation
        '''
        obs = []
        for item in obs_items:
            n_ = par.obs_dict[item][0] # number of items
            m_ = par.obs_dict[item][1] # normalization value
            if n_ == 1:
                v_ = self.info.get(item, 0)
                obs.append(min(1.0, v_/m_))
            else:
                values = self.info.get(item, [0]*n_)
                for v_ in values:
                    obs.append(min(1.0, v_/m_))
        return obs

    def set_ext_agent(self, ext_agent):
        '''
        defines the node state and params controlled by an external agent
        '''
        if ext_agent > -1:
            self.ext_agent = self.agents[ext_agent]
            self.reference_states = self.ext_agent.states
        else:
            self.ext_agent = None

    def reset(self):
        '''
        gymnasium interface method
        '''
        if not self.soft_reset:
            self.info = self.node.reset()
        if self.ext_agent:
            obs_items = self.ext_agent.obs_items
            obs = self.get_obs(obs_items)
            obs = np.array(obs, dtype=float)
            return obs, self.info
        return self.obs, self.info

    def update_action(self, a_id):
        '''
        the specified agent learns and decides (its part) of the control variables
        '''
        agent = self.agents[a_id]
        a_mask = agent.a_mask
        obs_items = agent.obs_items
        obs = self.get_obs(obs_items)
        a = agent.get_action(obs, self.reward, self.info, self.action)
        self.action[a_mask] = a

    def single_step(self):
        '''
        this method runs the system for a single step using the registered agents
        '''
        # observe state
        state = self.info['state']
        for agent_id in self.agents_ids[state]: # a state may have more than one agent
            self.update_action(agent_id)
        # now we can apply the action to the system
        self.reward, _, self.info = self.node.step(self.action)

    def run(self, steps):
        '''
        this method runs the system for a number of system steps using the registered agents
        '''
        for n in range(steps):
            self.single_step()
            ### DEBUG ###
            if DEBUG and n%100 == 0:
                print('step {}: t = {}, reward = {}, info = {}'.format(n,self.info['time'], self.reward, self.info))

    def run_agent_steps(self, agent_id, steps):
        '''
        this method runs the system for a number of system steps of one of the agents
        '''
        agent = self.agents[agent_id]
        while agent.total_steps < steps:
            self.single_step()

    def run_until(self, time):
        '''
        this method runs the system until a given time horizon using the registered agents
        '''
        t = 0
        while t < time:
            self.single_step()
            t = self.info['time']
            ### DEBUG ###
            if DEBUG and t%10_000 == 0:
                print('t = {}, reward = {}, info = {}'.format(self.info['time'], self.reward,self.info))

    def run_until_check(self, time, check_period, file_path):
        '''
        this method runs the system until a given time horizon using the registered agents
        and ckecks if the simulation is allowed to continue based on an external flag file
        '''
        t = 0
        steps = 0
        while t < time:
            self.single_step()
            steps += 1
            t = self.info['time']
            if steps % check_period == 0:
                if not read_flag_from_file(file_path):
                    break

    def run_double_check(self, time, check_period, file_path, second_check_p, check_cond = ('dep_ratio', 0.6)):
        '''
        this method runs the system until a given time horizon using the registered agents
        and performs two ckecks:
        1) if the simulation is allowed to continue based on an external flag file
        2) if a given condition is hold
        '''
        t = 0
        steps = 0
        while t < time:
            self.single_step()
            steps += 1
            t = self.info['time']
            if steps % check_period == 0:
                if not read_flag_from_file(file_path):
                    return t
            if steps % second_check_p == 0:
                observation = self.info[check_cond[0]]
                if observation < check_cond[1]:
                    return t
        return t

    def step(self, a):
        '''
        open ai gym interface method
        it is used by an external agent to make decisions on some control variables
        '''
        next_agent_id = self.ext_agent.next
        a_mask = self.ext_agent.a_mask
        self.action[a_mask] = a # the action provided is limited to some control variables

        # next agent in the list
        while next_agent_id > 0:
            self.update_action(next_agent_id)
            next_agent_id = self.agents[next_agent_id].next

        # apply the action to the system
        self.reward, _, self.info = self.node.step(self.action)

        # advance to the next step of the external agent
        self.to_next_step()
        
        # now we can let the external agent know what happened
        # the obs_items vector determines what the external agent observes
        obs_items = self.ext_agent.obs_items
        obs = self.get_obs(obs_items)
        obs = np.array(obs, dtype=float)

        return obs, self.reward, False, False, self.info
    
    def to_next_step(self):
        '''
        advances to the next step at which the external agent has to select an action
        '''
        # who is the external agent
        id = self.ext_agent.id

        # observe state
        state = self.info['state']

        # run the system with the agents in charge of other states
        while state not in self.reference_states:
            for agent_id in self.agents_ids[state]: # each state has one or more agents
                self.update_action(agent_id)
            
            # now we can apply the action to the system
            self.reward, _, self.info = self.node.step(self.action)

            # update state
            state = self.info['state']

        # we have reached a reference state
        # some agents may act before
        for agent_id in self.agents_ids[state]: # each state may have one or more agents
            if agent_id == id:
                break
            self.update_action(agent_id)
