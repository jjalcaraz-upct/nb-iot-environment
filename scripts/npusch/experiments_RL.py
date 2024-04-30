#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running experiments with a single RL agent in charge
of controlling the UE id, Imcs and Nrep.

Created on Nov 11, 2022

@author: juanjosealcaraz

"""

from numpy.random import default_rng
from numpy import savez
from scenarios import scenarios
from system.system_creator import create_system
from control_agents import DummyAgent
from controller import Controller
from wrappers import BasicSchedulerWrapper, DiscreteActions
import os
import gymnasium as gym
import concurrent.futures as cf
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
algorithms = {
'PPO2':PPO, 
'A2C':A2C,
'DQN': DQN
}


STEPS = 300_000 # 200_000
RUNS = 30
PARALLEL = True
PROCESSES = 30
algo_list = ['PPO2', 'A2C', 'DQN']

run_list = list(range(RUNS))

agent_0 = {
    'id': 0, # index of the agent
    'action_items': ['id', 'Imcs', 'Nrep'], # action items controlled by this agent
    'obs_items': ['total_ues', 'connection_time', 'loss', 'sinr', 'buffer', 'carrier_state'], # state indexes observed by this agent
    'next': 1, # next agent operating in the same nodeb state
    'states': ['Scheduling'] # nodeb state where this agent operates 
    }

agent_1 = {
    'id': 1, # carrier, delay and subcarriers
    'action_items': ['carrier', 'delay', 'sc'],
    'obs_items': [],
    'next': -1,
    'states': ['Scheduling']
    }

agent_2 = {
    'id': 2, # ce_level selection
    'action_items': ['carrier', 'ce_level', 'rar_Imcs', 'delay', 'sc', 'Nrep'],
    'obs_items': [],
    'next': -1,
    'states': ['RAR_window']
}

agent_3 = {
    'id': 3, # backoff selection
    'action_items': ['backoff'],
    'obs_items': [],
    'next': -1,
    'states': ['RAR_window_end'],
}

agent_4 = {
    'id': 4, # NPRACH configuration
    'action_items': ['th_C1', 'th_C0', 'sc_C0', 'sc_C1', 'sc_C2', 'period_C0', 'period_C1', 'period_C2'],
    'obs_items': [],
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
            DummyAgent(agent_0),
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

