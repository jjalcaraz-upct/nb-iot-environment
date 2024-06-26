#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running experiments with two agents: An online learning agent in charge 
of selecting Imcs and Nrep, and an RL agent in charge of selecting the UE id.

Created on Nov 11, 2022

@author: juanjosealcaraz

"""

from numpy.random import default_rng
from numpy import savez
from scenarios import scenarios
from system.system_creator import create_system
from control_agents import DummyAgent, RandomUserAgent, agents_conf
from agent_npusch import OnlineClassifierAgent
from controller import Controller
from wrappers import BasicWrapper
import concurrent.futures as cf
import os
import gymnasium as gym 
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
algorithms = {
'PPO2':PPO, 
'A2C':A2C,
'DQN': DQN
}

STEPS_1 = 1000 # 10_000
STEPS_2 = 3000 # 30_000
RUNS = 2
PARALLEL = True
PROCESSES = 2
algo_list = ['DQN']
SCENARIO = '2000_10_B'

run_list = list(range(RUNS))

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
        # mcs_agent.deactivate_learning()

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

