#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the controller class that orchestrates the operation of multiple control agents

Created on June 2022

@author: juanjosealcaraz

"""

import numpy as np
from gym import spaces
from system.parameters import *
from data_agents import DummyAgent, agents_conf

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
        self.action = np.array([0]*N_actions, dtype=int)
        self.obs = np.array([0]*state_dim, dtype=float)
        self.soft_reset = False
        self.reward = 0.0
        self.info = {}
        self.set_ext_agent(ext_agent)
        self._set_agents_ids()

    def _set_agents_ids(self):
        '''
        private method
        '''
        self.agents_ids = {}
        for agent in self.agents:
            states = agent.states
            for state in states:
                if state in self.agents_ids.keys():
                    self.agents_ids[state].append(agent.id)
                else:
                    self.agents_ids[state] = [agent.id]
        print('>> CONTROLLER AGENT REGISTER')
        print(self.agents_ids)

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
            space = spaces.MultiDiscrete(np.array(a_max))
        else:
            space = spaces.Discrete(a_max[0])
        return space

    def get_obs_space(self):
        '''
        for open ai gym compatibility
        '''
        s_mask = self.ext_agent.s_mask
        space = spaces.Box(low=0.0, high=1.0, shape=(len(s_mask),), dtype=np.float32)
        return space

    def set_ext_agent(self, ext_agent):
        '''
        defines the node state and params controlled by an external agent
        '''
        if ext_agent > -1:
            self.ext_agent = self.agents[ext_agent]
            self.reference_states = self.ext_agent.states
            if DEBUG:
                for agent in self.agents:
                    print(f'agent id {agent.id}, reference states {agent.states}') 
                state = self.info.get('state','idle')
                print('')
                print(f'current state {state}')
                print('')
        else:
            self.ext_agent = None


    def reset(self):
        '''
        open ai interface method
        '''
        if not self.soft_reset:
            self.obs, self.info = self.node.reset()
        if self.ext_agent:
            s_mask = self.ext_agent.s_mask
            return self.obs[s_mask], self.info
        return self.obs, self.info

    def update_action(self, a_id):
        '''
        the specified agent learns and decides (its part) of the control variables
        '''
        agent = self.agents[a_id]
        a_mask = agent.a_mask
        s_mask = agent.s_mask
        obs = self.obs[s_mask]
        a = agent.get_action(obs, self.reward, self.info, self.action)
        if DEBUG:
            print(f'update action a: {a}')
            print(f'a_mask: {a_mask}')
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
        self.obs, self.reward, _, self.info = self.node.step(self.action)

    def run(self, steps):
        '''
        this method runs the system for a number of system steps using the registered agents
        '''
        for n in range(steps):
            self.single_step()
            ### DEBUG ###
            if DEBUG and n%100 == 0:
                print('step {}: t = {}, reward = {}, info = {}'.format(n,self.info['time'], self.reward,self.info))

    
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
        self.obs, self.reward, _, self.info = self.node.step(self.action)

        # advance to the next step of the external agent
        self.to_next_step()
        
        # now we can let the external agent know what happened
        # here we filter what is returned to the agent
        # specific functions could be provided for this in future versions
        s_mask = self.ext_agent.s_mask
        obs = self.obs[s_mask]

        return obs, self.reward, False, self.info
    
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
            self.obs, self.reward, _, self.info = self.node.step(self.action)

            # update state
            state = self.info['state']

        # we have reached a reference state
        # some agents may act before
        for agent_id in self.agents_ids[state]: # each state may have one or more agents
            if agent_id == id:
                break
            self.update_action(agent_id)
