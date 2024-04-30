#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running experiments with two RL agents: one is in charge 
of selecting Imcs and Nrep, and the other is in charge of selecting the UE id.

Created on Nov 11, 2022

@author: juanjosealcaraz

"""

from system.parameters import set_global_parameters
set_global_parameters(N = 4, H = 40, Nc = 1)

from numpy.random import default_rng
from numpy import savez
from scenarios import scenarios
from system.system_creator import create_system
from data_agents import DummyAgent, RandomUserAgent, TrainedAgent, DiscreteTrainedAgent, agents_conf
from controller import Controller
from wrappers import BasicWrapper, BasicMcsWrapper, DiscreteActions
import concurrent.futures as cf
if 'stable_baselines3' in sys.modules:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    algorithms = {
    'PPO2':PPO, 
    'A2C':A2C,
    'DQN': DQN
    }
    print('stable-baselines 3 loaded')
else:
    from stable_baselines import DQN, A2C, PPO2
    from stable_baselines.common.cmd_util import make_vec_env
    algorithms = {
    'PPO2':PPO2, 
    'A2C':A2C,
    'DQN': DQN
    }
import os
import gym

STEPS_1 = 20_000
STEPS_2 = 50_000 # 20_000
RUNS = 30
PARALLEL = True
PROCESSES = 30
algo_list = ['PPO2']
# algo_list = ['PPO2', 'A2C', 'DQN']

# STEPS_1 = 10_000
# STEPS_2 = 5_000
# RUNS = 2
# PARALLEL = True
# PROCESSES = 2
# algo_list = ['ACER','TRPO','ACKTR']
# scenario = '2000_10_B'

run_list = list(range(RUNS))

class Evaluator():
    def __init__(self, scenario, algorithm):
        self.scenario = scenario
        self.algorithm = algorithm
        self.path = f'./results/{scenario}/2A_{algorithm}/'
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
        print(' > System created')
        
        # create agents
        agents = [
            RandomUserAgent(agents_conf[0], rng),
            DummyAgent(agents_conf[1]),
            DummyAgent(agents_conf[2]),
            DummyAgent(agents_conf[3]),
            DummyAgent(agents_conf[4]),
            DummyAgent(agents_conf[5])
        ]
        print(' > Agents created')

        # configure controller
        controller = Controller(node, agents = agents)
        _ = controller.reset()
        controller.set_ext_agent(1)
        print(' > Controller configured')

        # create the gym environment
        nbiot_env = gym.make('gym_system:System-v1', system = controller)
        print(' > gym environment created for mcs')

        # wrap the environment
        nbiot_env = BasicMcsWrapper(nbiot_env, verbose = True, n_report = 1_000)
        if self.algorithm in ['DQN',' TRPO', 'ACKTR', 'ACER']:
            nbiot_env = DiscreteActions(nbiot_env)
        print(' > Environment wrapped for mcs')

        # prepare the agent
        env = make_vec_env(lambda: nbiot_env, n_envs=1)
        print(' > Vectorised environment created')

        # select the agent's algorithm
        algo = algorithms[self.algorithm]

        # create the agent 
        rl_seed = i + 1000
        model = algo('MlpPolicy', env, verbose=0, seed = rl_seed)
        print(' > Model created')

        # and learn
        model.learn(total_timesteps = STEPS_1)
        print(' > Learning mcs completed!')

        # record the number of served users so far
        served_users_1 = perf_monitor.total_departures()

        # pack the trained agent       
        if self.algorithm in ['DQN',' TRPO', 'ACKTR', 'ACER']:
            rl_agent = DiscreteTrainedAgent(agents_conf[1], model, nbiot_env.original_action_space.nvec, deterministic = True)
        else:
            rl_agent = TrainedAgent(agents_conf[1], model, deterministic = True)
        print(' > mcs trained agent created')

        # and insert it in the controller
        controller.replace_agent(rl_agent, 1)
        print(' > mcs agent inserted')

        # now we will learn how to select the actions of agent 0 (ue id selection)
        controller.set_ext_agent(0)
        print(' > Controller reconfigured for external ue selection')

        # move to the next step
        controller.to_next_step()
        print(' > Controller advanced to next step')

        # create a new gym environment
        controller.soft_reset = True # to avoid reseting the node b
        nbiot_env = gym.make('gym_system:System-v1', system = controller)
        print(' > gym environment created for ue selection')

        # wrap the environment
        nbiot_env = BasicWrapper(nbiot_env, verbose = True, n_report = 1_000)
        print(' > Environment wrapped for ue selection')

        # prepare the agent
        env = make_vec_env(lambda: nbiot_env, n_envs=1)
        print(' > Vectorised environment created for ue selection')

        # select the agent's algorithm
        algo = algorithms[self.algorithm]

        # create the agent 
        rl_seed = i + 2000
        model = algo('MlpPolicy', env, verbose=0, seed = rl_seed)
        print(' > Model created')

        # and learn
        model.learn(total_timesteps = STEPS_2)
        print(' > Phase 2 completed!')

        # record the total number of served users
        served_users_2 = perf_monitor.total_departures()
        served_users = [served_users_1, served_users_2]

        # now save the results
        model_path = f'{self.path}/history_{STEPS_2}_{i}.npz'
        savez(model_path, delay = perf_monitor.delay_history, connection = perf_monitor.connection_history, served_users = served_users)

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

