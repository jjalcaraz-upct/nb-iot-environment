#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running experiments with a single RL agent in charge
of controlling the UE id, Imcs and Nrep.

Created on Nov 11, 2022

@author: juanjosealcaraz

"""

from parameters import set_global_parameters
set_global_parameters(N = 4, H = 40, Nc = 1)

from numpy.random import default_rng
from numpy import savez
from scenarios import scenarios
from test_utils import create_system
from agents import DummyAgent, scheduling_indexes, ce_selection_indexes, nprach_indexes, nprach_actions
from controller import Controller
from wrappers import BasicSchedulerWrapper, DiscreteActions
import concurrent.futures as cf
from stable_baselines import ACER, ACKTR, DQN, A2C, PPO2
from stable_baselines.common.cmd_util import make_vec_env
import os
import gym

STEPS = 300_000
RUNS = 30
PARALLEL = True
PROCESSES = 30
algo_list = ['PPO2', 'A2C', 'DQN']

run_list = list(range(RUNS))

algorithms = {
    'ACER': ACER,
    'ACKTR': ACKTR,
    'PPO2':PPO2, 
    'A2C':A2C,
    'DQN': DQN
}

# external agent:
ext_agent_conf = {
    'id': 0, # UE Imcs and Nrep selection
    'action_items': ['id', 'Imcs', 'Nrep'], # action items controlled by this agent
    's_mask': scheduling_indexes, # state indexes observed by this agent
    'next': 1, # next agent operating in the same nodeb state
    'states': ['Scheduling'] # nodeb state where this agent operates 
    }

agent_1 = {
    'id': 1, # carrier, delay and subcarriers
    'action_items': ['carrier', 'delay', 'sc'],
    's_mask': scheduling_indexes,
    'next': -1,
    'states': ['Scheduling']
    }

agent_2 = {
    'id': 2, # ce_level selection
    'action_items': ['carrier', 'ce_level', 'rar_Imcs', 'delay', 'sc', 'Nrep'],
    's_mask': ce_selection_indexes,
    'next': -1,
    'states': ['RAR_window']
}

agent_3 = {
    'id': 3, # backoff selection
    'action_items': ['backoff'],
    's_mask': nprach_indexes,
    'next': -1,
    'states': ['RAR_window_end'],
}

agent_4 = {
    'id': 4, # NPRACH configuration
    'action_items': nprach_actions,
    's_mask': nprach_indexes,
    'next': -1,
    'states': ['NPRACH_update']
}

class Evaluator():
    def __init__(self, scenario, algorithm):
        self.scenario = scenario
        self.algorithm = algorithm
        self.path = f'./results/{scenario}/1A_{algorithm}/'
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError:
                print (f'Creation of the directory {self.path} failed')
            else:
                print (f'Successfully created the directory {self.path}')
    
    def evaluate(self, i):
        # create random number generator
        rng = default_rng(seed = i)

        # extract configuration
        conf = scenarios[self.scenario]

        # create system
        node, perf_monitor, _, _ = create_system(rng, conf)

        # create agents
        agents = [
            DummyAgent(ext_agent_conf),
            DummyAgent(agent_1),
            DummyAgent(agent_2),
            DummyAgent(agent_3),
            DummyAgent(agent_4)
        ]

        # configure controller
        controller = Controller(node, agents = agents)
        _ = controller.reset()
        controller.set_ext_agent(0)

        # create the gym environment
        nbiot_env = gym.make('gym_system:System-v1', system = controller)

        # wrap the environment
        nbiot_env = BasicSchedulerWrapper(nbiot_env, verbose = True, n_report = 1_000)
        if self.algorithm in ['DQN',' TRPO', 'ACKTR', 'ACER']:
            nbiot_env = DiscreteActions(nbiot_env)

        # prepare the agent
        env = make_vec_env(lambda: nbiot_env, n_envs=1)

        # select the agent's algorithm
        algo = algorithms[self.algorithm]

        # create the agent 
        rl_seed = i + 1000
        model = algo('MlpPolicy', env, verbose=0, seed = rl_seed)

        # and learn
        model.learn(total_timesteps = STEPS)

        # now save the results
        model_path = f'{self.path}/history_{STEPS}_{i}.npz'
        savez(model_path, delay = perf_monitor.delay_history, connection = perf_monitor.connection_history)

def run(scenario):
    for alg in algo_list:
        evaluator = Evaluator(scenario, alg)
        with cf.ProcessPoolExecutor(PROCESSES) as E:
            results = E.map(evaluator.evaluate, run_list)

if __name__=='__main__':
    scenario = '2000_10_B'
    for alg in algo_list:
        evaluator = Evaluator(scenario, alg)
        if PARALLEL:
            with cf.ProcessPoolExecutor(PROCESSES) as E:
                results = E.map(evaluator.evaluate, run_list)
        else:
            for run in run_list:
                evaluator.evaluate(run)

