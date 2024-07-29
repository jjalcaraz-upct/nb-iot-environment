#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Multiple DQN agents

@author: juanjosealcaraz

"""
import system.parameters as par
from control_agents import DummyAgent
from agent_dqn import DQN_Agent
from itertools import product
import numpy as np

def to_discrete(dimensions):
    args = [range(d) for d in dimensions]  # List comprehension to generate ranges
    return [list(item) for item in product(*args)]  # Convert each tuple to a list

max_periods = [par.control_max_values['period_C0'][0], par.control_max_values['period_C1'][0], par.control_max_values['period_C1'][0]]
max_sc = [par.control_max_values['sc_C0'][0], par.control_max_values['sc_C1'][0], par.control_max_values['sc_C2'][0]]

# action items 'sc_C0', 'sc_C1', 'sc_C2', 'period_C0', 'period_C1', 'period_C2'

class NPRACH_DQN_Agent(DummyAgent):
    '''
    agent that controls NPRACH parameters with one DQN agent pero CE level
    '''
    def __init__(self, dict, env, metrics, total_timesteps, seed = 123):
        super().__init__(dict)
        self.env = env
        self.samples = {metric: [] for metric in metrics}
        self.obs_dim = env.observation_space.shape[0]
        self.actions = to_discrete([max_sc[0], max_periods[0] - 1])
        self.num_actions = len(self.actions)

        obs_shape = (self.obs_dim,)

        self.agent_C0 = DQN_Agent(obs_shape, 
                                   self.num_actions, 
                                   total_timesteps = total_timesteps,
                                   seed = seed)
        
        obs_shape = (self.obs_dim + 2,)

        self.agent_C1 = DQN_Agent(obs_shape, 
                                   self.num_actions, 
                                   total_timesteps = total_timesteps,
                                   seed = seed)
        
        obs_shape = (self.obs_dim + 4,)

        self.agent_C2 = DQN_Agent(obs_shape, 
                                   self.num_actions, 
                                   total_timesteps = total_timesteps,
                                   seed = seed)
        
        self.total_timesteps = total_timesteps
    

    def get_action(self, obs):
        a0 = self.agent_C0.get_action(obs)
        conf_C0 = self.actions[a0]
        obs_1 = np.concatenate((obs, np.array(conf_C0)), axis=0)
        a1 = self.agent_C1.get_action(obs_1)
        conf_C1 = self.actions[a1]
        obs_2 = np.concatenate((obs_1, np.array(conf_C1)), axis=0)
        a2 = self.agent_C2.get_action(obs_2)
        conf_C2 = self.actions[a2]
        action = [conf_C0[0], conf_C1[0], conf_C2[0], conf_C0[1] + 1, conf_C1[1] + 1, conf_C2[1] + 1]
        return action, [np.array([a0]), np.array([a1]), np.array([a2])], obs_1, obs_2
    
    def learn(self):
        obs, _ = self.env.reset()
        action, a_list, obs_1, obs_2 = self.get_action(obs)

        for t in range(self.total_timesteps):
            next_obs, reward, _, _, info = self.env.step(action)
            next_action, next_a_list, next_obs_1, next_obs_2 = self.get_action(next_obs)  
            self.agent_C0.store_transition(obs, next_obs, a_list[0], reward)
            self.agent_C1.store_transition(obs_1, next_obs_1, a_list[1], reward)
            self.agent_C2.store_transition(obs_2, next_obs_2, a_list[2], reward)
            action = next_action
            a_list = next_a_list
            obs = next_obs
            obs_1 = next_obs_1
            obs_2 = next_obs_2

            # store samples from previous step
            for metric, samples in self.samples.items():
                if metric in ['th_C0', 'th_C1', 'arrivals_C0', 'arrivals_C1', 'arrivals_C2', 'colisions_C0', 'colisions_C1', 'colisions_C2', 'detections_C0', 'detections_C1', 'detections_C2']:
                    continue
                sample = info.get(metric, None) # if the metric is not in info, the metric is the margin (beta) factor
                if metric == 'NPRACH_detection':
                    detections_av = [np.mean(inner_list) for inner_list in sample]
                    samples.append(sum(detections_av))
                elif metric == 'NPRACH_collision':
                    collisions_av = [np.mean(inner_list) for inner_list in sample]
                    samples.append(sum(collisions_av))    
                elif isinstance(sample, list):
                    # Avoid division by zero for empty lists
                    avg = sum(sample) / len(sample) if sample else 0
                    samples.append(avg)
                else:
                    # If the value is not a list, store it as is
                    samples.append(sample)

            if 'arrivals_C0' in self.samples.keys():
                # estimate arrivals per CE level
                self.samples['arrivals_C0'].append(detections_av[0]+collisions_av[0])
                self.samples['arrivals_C1'].append(detections_av[1]+collisions_av[1])
                self.samples['arrivals_C2'].append(detections_av[2]+collisions_av[2])

            if 'colisions_C0' in self.samples.keys():
                collisions = info.get('colision_ratios', [0, 0, 0])
                # estimate collision rations per CE level
                self.samples['colisions_C0'].append(collisions[0])
                self.samples['colisions_C1'].append(collisions[1])
                self.samples['colisions_C2'].append(collisions[2])
            
            if 'detections_C0' in self.samples.keys():
                detections = info.get('detection_ratios', [0, 0, 0])
                # estimate arrivals per CE level
                self.samples['detections_C0'].append(detections[0])
                self.samples['detections_C1'].append(detections[1])
                self.samples['detections_C2'].append(detections[2])
        

