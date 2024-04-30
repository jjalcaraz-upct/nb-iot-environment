#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for NPRACH configuration experiments

Created on December 2023

@author: juanjosealcaraz

RL controls the multiplicative factor instead of UCB

"""
import numpy as np
from numpy import savez

from system.system_creator import create_system
from numpy.random import default_rng
from controller import Controller
from control_agents import DummyAgent
from agent_nprach import NPRACH_THAgent

import gymnasium as gym

import concurrent.futures as cf

RUNS = 20
run_list = list(range(RUNS))
PROCESSES = 4
STEPS = 100_000
VERBOSE = False
USERS = 1000
scenario_list = ['mixed_05', 'uniform', 'random']

# scenario_list = ['mixed_th_05', 'mixed_th_08', 'uniform_th']
ratios = {'uniform': 1.0,
          'mixed_05': 0.5,
          'random': 0.5}

algo_list = ['PPO']

discretize = {'A2C': True,
              'PPO': True,
              'DQN': True,
              'SAC': False,
              'TD3': False,
              'DDPG': False}

metrics = ['departures', 'NPRACH_occupation', 'service_times', 'beta']

FINAL_RESULTS_PATH = './results/'

def append_to_file(filename, line):
    with open(filename, "a") as file:
        file.write(line)

# agent configurations:
agent_0 = {
    'id': 0, # UE Imcs and Nrep selection
    'action_items': ['id', 'Imcs', 'Nrep', 'carrier', 'delay', 'sc'], # action items controlled by this agent
    # 'obs_items': ['total_ues', 'connection_time', 'loss', 'sinr', 'buffer', 'carrier_state'], # state indexes observed by this agent
    'obs_items': [],
    'next': -1, # next agent operating in the same nodeb state
    'states': ['Scheduling'] # nodeb state where this agent operates 
    }

agent_1 = {
    'id': 1, # ce_level selection
    'action_items': ['ce_level', 'rar_Imcs', 'Nrep'],
    'obs_items': [],
    'next': -1,
    'states': ['RAR_window']
}

agent_2 = {
    'id': 2, # RA parameters selection
    'action_items': ['rar_window', 'mac_timer', 'transmax', 'panchor', 'backoff'],
    'obs_items': [],
    'next': -1,
    'states': ['RAR_window_end'],
}

agent_3 = {
    'id': 3, # NPRACH configuration
    'action_items': ['th_C1', 'th_C0', 'sc_C0', 'sc_C1', 'sc_C2', 'period_C0', 'period_C1', 'period_C2'],
    'obs_items': ['detection_ratios', 'colision_ratios', 'msg3_detection', 'NPRACH_occupation', 'av_delay', 'distribution'],
    'next': -1,
    'states': ['NPRACH_update']
}

class Evaluator():
    def __init__(self, scenario, algorithm, model_path = None, logfile = None):
        self.scenario = scenario
        self.algorithm = algorithm
        self.logfile = logfile
        self.model_path = model_path
        self.ratio = ratios[scenario]
        self.random = scenario == 'random'
    
    def evaluate(self, i):
        # import required libraries
        if self.algorithm == 'SAC':
            from stable_baselines3 import SAC
            algo = SAC
        elif self.algorithm == 'PPO':
            from stable_baselines3 import PPO
            algo = PPO
        elif self.algorithm == 'A2C':
            from stable_baselines3 import A2C
            algo = A2C
        elif self.algorithm == 'DQN':
            from stable_baselines3 import DQN
            algo = DQN
        elif self.algorithm == 'TD3':
            from stable_baselines3 import TD3
            algo = TD3
        elif self.algorithm == 'DDPG':
            from stable_baselines3 import DDPG
            algo = DDPG               
        from stable_baselines3.common.env_util import make_vec_env
        from wrappers import NPRACH_agent_wrapper
        import torch as th
        
        th.set_num_threads(1)

        # create random number generator
        rng = default_rng(seed = i)

        # simulator configuration
        conf = {
            'ratio': self.ratio, # ratio of uniform/beta traffic
            'M': USERS, # number of UEs
            'buffer_range': [100, 600], # range for the number of bits in the UE buffer
            'reward_criteria': 'throughput', # users served
            'random': self.random
            }

        # create system
        node, _, _, _ = create_system(rng, conf)

        # create agents
        # agents are arranged in a list ordered by their id attribute
        agents = [
            DummyAgent(agent_0),
            DummyAgent(agent_1),
            DummyAgent(agent_2),
            DummyAgent(agent_3)
        ]

        # configure controller
        controller = Controller(node, agents = agents)
        _ = controller.reset()
        controller.set_ext_agent(3)

        # create the intermediate agent
        agent = NPRACH_THAgent(agent_3, metrics, verbose = False, no_th = False)
        # create the gym environment
        nbiot_env = gym.make('gym_system:System-v1', system = controller)
        
        # wrap the environment with the intermediate agent
        discrete = discretize[self.algorithm]
        norm = 100
        observation = [0,1,2,3,4,5,9,10,11,12]
        nbiot_env = NPRACH_agent_wrapper(nbiot_env, agent, n_actions = 26, obs_items = observation, obs_window = 1, bounds = [0.3, 2.0], discrete = discrete, norm = norm)

        # prepare the RL agent
        env = make_vec_env(lambda: nbiot_env, n_envs=1)

        # create the RL agent 
        rl_seed = i + 1000
        
        if self.algorithm in ['TD3', 'DDPG']:
            model = algo('MlpPolicy', env, verbose=0, seed = rl_seed, device = 'cpu', learning_starts = 1000, train_freq = (1, "step"))
        elif self.algorithm in ['A2C', 'PPO']:
            ent_coef = 0.01
            model = algo('MlpPolicy', env, verbose=0, ent_coef = ent_coef, seed = rl_seed, device = 'cpu')
        else: #SAC
            model = algo('MlpPolicy', env, verbose=0, seed = rl_seed, device = 'cpu')

        # and learn
        model.learn(total_timesteps = STEPS)

        # save model
        if self.model_path:
            model_file = self.model_path + f'model_MB_{alg}_{scenario}_{i}'
            # save model
            model.save(model_file)

        # get the results:
        samples = nbiot_env.get_wrapper_attr('samples')

        # close the environment
        nbiot_env.close()

        return samples

if __name__=='__main__':
    for scenario in scenario_list:

        for alg in algo_list:      
            # define results file  
            FINAL_RESULTS = FINAL_RESULTS_PATH + f'RL_MB_{alg}_{scenario}.npz'

            evaluator = Evaluator(scenario, alg)
            
            # run experiments
            with cf.ProcessPoolExecutor(PROCESSES) as E:
                results = E.map(evaluator.evaluate, run_list)
            
            samples_dict = {metric: [] for metric in metrics}
            for r_ in results:
                for metric in metrics:
                    samples_dict[metric].append(r_[metric])

            arrays_dict = {metric: np.array(samples_dict[metric]) for metric in metrics}

            savez(FINAL_RESULTS, **arrays_dict)
