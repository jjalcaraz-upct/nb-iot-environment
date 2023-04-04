#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running experiments with two agents: An online learning agent in charge 
of selecting Imcs and Nrep, and an RL agent in charge of selecting the UE id.

Created on Nov 11, 2022

@author: juanjosealcaraz

"""

from parameters import set_global_parameters
set_global_parameters(N = 4, H = 40, Nc = 1)

from numpy.random import default_rng
from numpy import savez
from scenarios import scenarios
from test_utils import create_system
from agents import DummyAgent, RandomUserAgent, OnlineClassifierAgent, agents_conf
from controller import Controller
from wrappers import BasicWrapper, DiscreteActions
import concurrent.futures as cf
from stable_baselines import ACER, ACKTR, DQN, A2C, PPO2
from stable_baselines.common.cmd_util import make_vec_env
import os
import gym

STEPS_1 = 10_000 
STEPS_2 = 50_000 # 30_000
RUNS = 30
PARALLEL = True
PROCESSES = 30
algo_list = ['PPO2', 'A2C', 'DQN']
SCENARIO = '2000_10_B'

run_list = list(range(RUNS))

algorithms = {
    'ACER': ACER,
    'ACKTR': ACKTR,
    'PPO2':PPO2, 
    'A2C':A2C,
    'DQN': DQN
}

class Evaluator():
    def __init__(self, scenario, algorithm):
        self.scenario = scenario
        self.algorithm = algorithm
        self.path = f'./results/{scenario}/OL_{algorithm}/'
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

        # online classifier agent
        mcs_agent = OnlineClassifierAgent(agents_conf[1], rng)
        
        # create agents
        agents = [
            RandomUserAgent(agents_conf[0], rng),
            mcs_agent,
            DummyAgent(agents_conf[2]),
            DummyAgent(agents_conf[3]),
            DummyAgent(agents_conf[4]),
            DummyAgent(agents_conf[5])
        ]

        # configure controller
        controller = Controller(node, agents = agents)
        _ = controller.reset()
        # controller.run_until(STEPS_1)
        controller.run(STEPS_1)

        # deactivate mcs learning
        mcs_agent.deactivate_learning()

        # record the number of served users so far
        served_users_1 = perf_monitor.total_departures()

        # reconfigure controller
        controller.set_ext_agent(0)

        # move to the next step
        controller.to_next_step()

        # create the gym environment
        controller.soft_reset = True # to avoid reseting the node b
        nbiot_env = gym.make('gym_system:System-v1', system = controller)

        # wrap the environment
        nbiot_env = BasicWrapper(nbiot_env, verbose = True, n_report = 1_000)

        # prepare the agent
        env = make_vec_env(lambda: nbiot_env, n_envs=1)

        # select the agent's algorithm
        algo = algorithms[self.algorithm]

        # create the agent 
        rl_seed = i + 1000
        model = algo('MlpPolicy', env, verbose=0, seed = rl_seed)

        # and learn
        model.learn(total_timesteps = STEPS_2)

        # record the total number of served users
        served_users_2 = perf_monitor.total_departures()
        served_users = [served_users_1, served_users_2]

        # now save the results
        model_path = f'{self.path}/history_{STEPS_2}_{i}.npz'
        savez(model_path, delay = perf_monitor.delay_history, connection = perf_monitor.connection_history, served_users = served_users)
        # node.brief_report()

def run(scenario):
    for alg in algo_list:
        evaluator = Evaluator(scenario, alg)
        with cf.ProcessPoolExecutor(PROCESSES) as E:
            results = E.map(evaluator.evaluate, run_list)


if __name__=='__main__':
    for alg in algo_list:
        evaluator = Evaluator(SCENARIO, alg)
        if PARALLEL:
            with cf.ProcessPoolExecutor(PROCESSES) as E:
                results = E.map(evaluator.evaluate, run_list)
        else:
            for run in run_list:
                evaluator.evaluate(run)

