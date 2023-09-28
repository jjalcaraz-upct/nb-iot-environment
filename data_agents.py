#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Definition of several types of control agents

Created on Oct 2022

@author: juanjosealcaraz

"""
# import time
import numpy as np
from skmultiflow.trees import HoeffdingTreeClassifier
from itertools import product
import matplotlib.pyplot as plt
from wrappers import to_discrete
from system.utils import find_next_integer, find_next_integer_index
from system.parameters import *

DEBUG_0 = False
    
# configuration of the agents that control the system
# the names and indexes of the state and control elements
# are defined in parameter.py

nprach_actions = ['rar_window', 'mac_timer', 'transmax', 'panchor', 
                    'period_C0', 'rep_C0', 'sc_C0',
                    'period_C1', 'rep_C1', 'sc_C1',
                    'period_C2', 'rep_C2', 'sc_C2']

agents_conf = [
    {'id': 0, # UE selection
    'action_items': ['id'], # action items controlled by this agent
    's_mask': scheduling_indexes, # state indexes observed by this agent
    'next': 1, # next agent operating in the same nodeb state
    'states': ['Scheduling'] # nodeb state where this agent operates 
    },
    
    {'id': 1, # Imcs, N_rep, N_ru selection
    'action_items': ['Imcs', 'Nrep', 'N_ru'],
    's_mask': scheduling_indexes,
    'next': 2,
    'states': ['Scheduling']
    },

    {'id': 2, # carrier, delay and subcarriers
    'action_items': ['carrier', 'delay', 'sc'],
    's_mask': scheduling_indexes,
    'next': -1,
    'states': ['Scheduling']
    },

    {'id': 3, # ce_level selection
    'action_items': ['carrier', 'ce_level', 'rar_Imcs', 'delay', 'sc', 'Nrep'],
    's_mask': ce_selection_indexes,
    'next': -1,
    'states': ['RAR_window']
    },

    {'id': 4, # backoff selection
    'action_items': ['backoff'],
    's_mask': nprach_indexes,
    'next': -1,
    'states': ['RAR_window_end'],
    },

    {'id': 5, # NPRACH configuration
    'action_items': nprach_actions,
    's_mask': nprach_indexes,
    'next': -1,
    'states': ['NPRACH_update']
    }
]

# auxiliary functions for retrieving action indexes and max action values
def get_control_indexes(name_list):
    return [control_items[name] for name in name_list]

def get_max_control_values(name_list):
    return [control_max_values[name][N_carriers - 1] for name in name_list]

def get_control_default_values(name_list):
    return [control_default_values[name] for name in name_list]

class DummyAgent:
    '''dummy agent that simply applies a fixed action'''
    def __init__(self, dict):
        self.__dict__.update(dict)
        action_items = self.action_items
        self.a_mask = get_control_indexes(action_items)
        self.a_max = get_max_control_values(action_items)
        self.fixed_action = get_control_default_values(action_items)

    def set_action(self, fixed_action):
        self.fixed_action = fixed_action

    def get_action(self, obs, r, info, action):
        # action contains the action so far
        # agents communicate using this argument
        return self.fixed_action

    def print_action(self):
        for name, value in zip(self.action_items, self.fixed_action):
            print(f' {name}: {value}')
        print('')


class TrainedAgent(DummyAgent):
    def __init__(self, dict, model, deterministic = False):
        super().__init__(dict)
        self.model = model
        self.deterministic = deterministic

    def get_action(self, obs, r, info, action):
        action, _ = self.model.predict(obs, deterministic = self.deterministic)
        return action


class DiscreteTrainedAgent(TrainedAgent):
    '''
    auxiliary class that encapsulates an rl agent that selects discrete actions but interacts with an envirominent with multidiscrete actions
    '''
    def __init__(self, dict, model, nvec, deterministic = False):
        super().__init__(dict, model, deterministic)
        self.actions = to_discrete(nvec)
    
    def get_action(self, obs, r, info, action):
        action, _ = self.model.predict(obs, deterministic = self.deterministic)
        return self.actions[action]


class RandomUserAgent(DummyAgent):
    '''agent that selects a user at random'''
    def __init__(self, dict, rng):
        super().__init__(dict)
        self.rng = rng
    
    def get_action(self, obs, r, info, action):
        users = min(len(info['ues']), N_users)
        ue_list = info['ues']
        if DEBUG_0:
            print(' ')
            print(f' SELECTION AGENT: ue_list = {ue_list}')
        if users == 0:
            if DEBUG_0:
                print(' SELECTION AGENT: selects 0')
                print(' ')
            return [0]
        selection = self.rng.integers(users)
        if DEBUG_0:
            print(f' SELECTION AGENT: selects {selection}')
            print(' ')
        return [selection]
    
# ############################# OL AGENT ###################################

# auxiliary structures for selectin I_mcs, I_rep and N_rep in OnlineClassifierAgent
I_MCS_MAX = 7
I_REP_MAX = 8
I_mcs_values = list(range(0,I_MCS_MAX))
I_rep_values = list(range(0,I_REP_MAX))

def get_confs(buffer):
    '''generates possible configurations for a given buffer'''
    confs = {}
    for i_mcs in I_mcs_values:
        i_tbs = Imcs_to_Itbs[i_mcs]
        lst = tbs_lists[i_tbs]
        tbs, i_nru = find_next_integer_index(lst, buffer)
        if tbs > buffer:
            N_ru = N_ru_list[i_nru]
            for i_rep in I_rep_values:
                confs[(i_mcs, i_rep, i_nru)] = N_ru * N_rep_list[i_rep]
    return confs

ordered_confs_dict = {} # configurations per buffer size ordered in increasing number of resources
confs_rus = {} # N_rus given i_nrep and i_nru
N_list = [] # list of N_rus
buffer_values = list(range(20,501,10)) # considered buffer sizes
for buffer in buffer_values:
    confs = get_confs(buffer)
    ordered_confs = []
    for key, N in sorted(confs.items(), key=lambda item: item[1]):
        ordered_confs.append(key)
        confs_rus[key] = N 
        if N not in N_list:
            N_list.append(N)
    ordered_confs_dict[buffer] = ordered_confs
N_list.sort() # increasing order

# auxiliary function for retrieving inputs from a dictionary of lists
def get_element(dict, key):
    l_ = dict.get(key)
    element = l_.pop(0)
    if not l_: # empty list
        dict.pop(key)
    return element

def fix_ue_i(i, info):
    # if the selected ue is beyond the max, then select ue 0
    if len(info['ues'])<=i:
        return 0 # by convention the nodeb chooses the first in the list
    return i

# required indexes of the control and the state
id_i = control_items['id']
sinr_i = user_items['sinr']
loss_i = user_items['loss']
buffer_i = user_items['buffer']
carrier_i = user_vars * N_users +1

class OnlineClassifierAgent(DummyAgent):
    '''agent that selects I_mcs, N_rep using an online learning algorithm'''
    def __init__(self, dict, rng, tie_threshold = 0.05, grace_period = 200, split_confidence=1e-07):
        super().__init__(dict)
        self.rng = rng
        self.classifier = HoeffdingTreeClassifier(tie_threshold = tie_threshold, grace_period = grace_period, split_confidence = split_confidence)
        self.predictions = {}
        self.harq_ues = {}
        self.inputs = {}
        self.hits = []
        self.fit_classifier = HoeffdingTreeClassifier(tie_threshold = tie_threshold, grace_period = grace_period, split_confidence = split_confidence)
        self.fit_predictions = {}
        self.fit_inputs = {}
        self.fit_hits = []
        self.action_method = self.get_action_learning
        self.sample_counter = 0
        # self.total_time = 0.0
        # self.total_calls = 0

    def get_action(self, obs, r, info, action):
        '''executes the configured method'''
        action = self.action_method(obs, r, info, action)
        return action
    
    # def get_avg_time(self):
    #     return self.total_time / self.total_calls

    def deactivate_learning(self):
        self.action_method = self.get_action_plain

    def action_required(self, sinr, info):
        if sinr > 0:
            return False
        ues = info.get('ues', [])
        if len(ues)==0:
            return False
        return True

    def get_action_plain(self, obs, r, info, action):
        '''
        Returns Imcs, N_rep, N_ru
        '''
        i = action[id_i] # gets the selected agent
        i = fix_ue_i(i, info)
        sinr = obs[sinr_i + user_vars * i]
        if not self.action_required(sinr, info): # in HARQ or when no UE
            if DEBUG_0:
                print(f' AGENT selected user {i}, normalized sinr: {sinr}, DUMMY ACTION')
            return [0, 0, 0] # dummy action
        else:
            loss = obs[loss_i + user_vars * i] # gets the loss of the selected UE
            buffer = obs[buffer_i + user_vars * i] # gets the buffer size of the selected UE
            b_, bi_ = find_next_integer_index(buffer_values, buffer)
            ordered_confs = ordered_confs_dict[b_]
            carrier_obs = list(obs[carrier_i:])
            sel = False
            # select action
            N_max = N_list[-1]
            for N in N_list: # determine N_rus that fit
                fit_input = [N] + carrier_obs
                fit_pred = self.fit_classifier.predict(np.array([fit_input]))
                if fit_pred == [0]:
                    N_max = N
                    break
            while not sel:
                for i, conf in enumerate(ordered_confs):
                    if confs_rus[conf] < N_max:
                        input = [conf[0], conf[1], conf[2], loss]
                        y_pred = self.classifier.predict(np.array([input]))
                        if y_pred == [1]:
                            sel = True
                            break
                    else:
                        break
                # try with less data
                bi_ -= 1
                if bi_ < 0:
                    break
                b_ = buffer_values[bi_] 
                ordered_confs = ordered_confs_dict[b_]
            
            if not sel:
                b_ = find_next_integer(buffer_values, buffer)
                ordered_confs = ordered_confs_dict[b_]
                conf = self.rng.choice(ordered_confs)        
            return conf


    def get_action_learning(self, obs, r, info, action):
        '''
        Returns Imcs, N_rep, N_ru
        '''
        i = action[id_i] # gets the selected agent
        i = fix_ue_i(i, info)
        if DEBUG_0:
            print(' AGENT observation: ')
            print(obs)
        sinr = obs[sinr_i + user_vars * i] # gets the sinr of this user
        sel = False
        if not self.action_required(sinr, info): # in HARQ or when no UE
            if len(info['ues'])>0:
                ue_id = info['ues'][i]
                self.harq_ues.setdefault(ue_id, []).append(ue_id)
                if DEBUG_0:
                    print(f' AGENT selected user {ue_id}, normalized sinr: {sinr}, DUMMY ACTION')
            conf = [0, 0, 0] # dummy action
        else:
            loss = obs[loss_i + user_vars * i] # gets the loss of the selected UE
            buffer = obs[buffer_i + user_vars * i] # gets the buffer size of the selected UE
            carrier_obs = list(obs[carrier_i:])
            ue_id = info['ues'][i]
            b_, bi_ = find_next_integer_index(buffer_values, buffer)
            ordered_confs = ordered_confs_dict[b_]
            if self.sample_counter < 2_000:
                for i_, conf in enumerate(ordered_confs):
                    N = confs_rus[conf]
                    fit_input = [N] + carrier_obs
                    fit_pred = self.fit_classifier.predict(np.array([fit_input]))
                    if fit_pred == [0]: # if it does not fit just pick the previous conf
                        conf = ordered_confs[max(0, i_-1)]
                        sel = True
                        break
                    input = [conf[0], conf[1], conf[2], loss]
                    y_pred = self.classifier.predict(np.array([input]))
                    if y_pred == [1]:
                        sel = True
                        break
            else:
                # determine N_rus that fit
                N_max = N_list[-1]
                for N in N_list:
                    fit_input = [N] + carrier_obs
                    fit_pred = self.fit_classifier.predict(np.array([fit_input]))
                    if fit_pred == [0]:
                        N_max = N
                        break
                while not sel: 
                    for i_ ,conf in enumerate(ordered_confs): # look for a configuration that works
                        if confs_rus[conf] < N_max:
                            input = [conf[0], conf[1], conf[2], loss]
                            y_pred = self.classifier.predict(np.array([input]))
                            if y_pred == [1]:
                                if DEBUG_0:
                                    print(f' AGENT predicts {conf} works')
                                    print(f' ')
                                sel = True
                                break
                        else:
                            break
                    # try with less data
                    bi_ -= 1
                    if bi_ < 0:
                        sel = True # all TBS exhausted, just pick one conf at random
                        c_ = self.rng.choice(ordered_confs[:max(1,i_)])
                        conf = (c_[0], c_[1], c_[2])
                        if DEBUG_0:
                            print(f' >>> random conf: {conf}')
                        break
                    b_ = buffer_values[bi_]
                    ordered_confs = ordered_confs_dict[b_]
                    if DEBUG_0:
                        print(f' >>> try with packet: {b_}')
            if not sel:
                # just pick one conf at random to explore
                sel = True
                b_ = find_next_integer(buffer_values, buffer)
                ordered_confs = ordered_confs_dict[b_]
                c_ = self.rng.choice(ordered_confs)
                conf = (c_[0], c_[1], c_[2])
                if DEBUG_0:
                    print(f' >>> random conf: {conf}')
            
            input = [conf[0], conf[1], conf[2], loss]
            y_pred = self.classifier.predict(np.array([input]))
            self.inputs.setdefault(ue_id, []).append(input)
            self.predictions.setdefault(ue_id, []).append(y_pred[0])

            # fit_input = [conf[0], conf[1], conf[2], b_] + carrier_obs
            if DEBUG_0:
                print(f' >> selected conf: {conf} for packet {b_} with buffer {buffer}')
            N = confs_rus[conf]
            fit_input = [N] + carrier_obs
            fit_pred = self.fit_classifier.predict(np.array([fit_input]))
            self.fit_inputs.setdefault(ue_id, []).append(fit_input)
            self.fit_predictions.setdefault(ue_id, []).append(fit_pred[0])

            if DEBUG_0:
                print(f' AGENT selects action {conf} for UE {ue_id}')
                print(f' ')

        if DEBUG_0:
            print(f' AGENT memory contains inputs: {list(self.inputs.keys())}')    

        # get the results of previous actions
        receptions = info['receptions']
        errors = info['errors']
        unfit = info['unfit']

        if DEBUG_0:
            print(f' AGENT receptions : {receptions}')
            print(f' AGENT errors : {errors}')
            print(f' AGENT unfits : {unfit}')

        for ue_id in unfit:
            if DEBUG_0:
                print(f' AGENT ue_id {ue_id} DOESNT FIT!!')
            if ue_id in self.harq_ues:
                _ = get_element(self.harq_ues, ue_id) # simply remove it 
            else:
                _ = get_element(self.inputs, ue_id) # remove it 
                self.update_fit_model(ue_id, 0) # and update fit model

        # update the model  
        for ue_id in receptions.keys():
            if DEBUG_0:
                print(f' AGENT ue_id {ue_id} NPUSCH OK')
            if ue_id in self.harq_ues:
                _ = get_element(self.harq_ues, ue_id) # simply remove it
            else:
                self.update_model(ue_id, 1)
                self.update_fit_model(ue_id, 1)
        
        for ue_id in errors:
            if DEBUG_0:
                print(f' AGENT ue_id {ue_id} NPUSCH ERROR!')
            if ue_id in self.harq_ues:
                _ = get_element(self.harq_ues, ue_id) # simply remove it
            else:
                self.update_model(ue_id, 0)
                self.update_fit_model(ue_id, 1)
        
        return conf

    def update_model(self, ue_id, label):
        inputs = self.inputs
        predictions = self.predictions
        classifier = self.classifier
        hits = self.hits

        if ue_id not in inputs: # Rtx packets are not in inputs
            return

        input = get_element(inputs, ue_id)
        i_mcs = input[0]
        i_nrep = input[1]
        i_nru = input[2]
        loss = input[3]
        X = []
        y = []
        if label == 0: # not rx (less nrep and more mcs would not be received either)
            nrep_list = list(range(i_nrep,-1,-1))
            mcs_list = list(range(i_mcs,I_MCS_MAX))
            for i_m, i_r in product(mcs_list, nrep_list):
                x = [i_m, i_r, i_nru, loss]
                X.append(x)
                y.append(0)
        else: # rx (more nrep and less mcs would be received too)
            nrep_list = list(range(i_nrep,I_REP_MAX))
            mcs_list = list(range(i_mcs,-1,-1))
            for i_m, i_r in product(mcs_list, nrep_list):
                x = [i_m, i_r, i_nru, loss]
                X.append(x)
                y.append(1)

        classifier.partial_fit(np.array(X), y)

        # log accuracy
        y_pred = get_element(predictions, ue_id)
        if label == y_pred:
            hits.append(1)
        else:
            hits.append(0)

    def update_fit_model(self, ue_id, label):
        self.sample_counter += 1
        inputs = self.fit_inputs
        predictions = self.fit_predictions
        classifier = self.fit_classifier
        hits = self.fit_hits

        if ue_id not in inputs: # Rtx packets are not in inputs
            return

        input = get_element(inputs, ue_id)
        n_ru = input[0]
        carrier_obs = input[1:]

        X = []
        y = []
        if label == 0: # not fit 
            for n_ in reversed(N_list):
                if n_ >= n_ru: # more RUs would not fit either
                    x = [n_] + carrier_obs
                    X.append(x)
                    y.append(0)
                else:
                    break
        else: # fits
            for n_ in N_list:
                if n_ <= n_ru: # fewer RUs would fit too
                    x = [n_] + carrier_obs
                    X.append(x)
                    y.append(1)
                else:
                    break
        
        classifier.partial_fit(np.array(X), y)

        # log accuracy
        y_pred = get_element(predictions, ue_id)
        if label == y_pred:
            hits.append(1)
        else:
            hits.append(0)

    def get_hit_accuracy(self):
        steps = len(self.hits)
        accuracy = [sum(self.hits[:i])/len(self.hits[:i]) for i in range(1, steps)]
        return accuracy
    
    def get_fit_accuracy(self):
        steps = len(self.fit_hits)
        accuracy = [sum(self.fit_hits[:i])/len(self.fit_hits[:i]) for i in range(1, steps)]
        return accuracy
    
    def plot_accuracy(self):
        steps = len(self.hits)
        time = [i for i in range(1, steps)]
        accuracy = self.get_hit_accuracy()
        plt.figure(1)
        plt.plot(time, accuracy)
        plt.grid()
        plt.savefig(f'bler_accuracy.png')

        steps = len(self.fit_hits)
        time = [i for i in range(1, steps)]
        accuracy = self.get_fit_accuracy()
        plt.figure(2)
        plt.plot(time, accuracy)
        plt.grid()
        plt.savefig(f'fit_accuracy.png')

# ########################## NBLA AGENT ##############################

losses = [80,90,100,110,120,130,135,140,200]
p_bits = [150,200,250,300,350,400,450,500]
    
MAX_I_REP = 2 # limit for the NBLAAgent for better performance

class NBLAAgent(DummyAgent):
    '''
    NarrowBand Link Adaptation agent adapted for a lookup table 
    indexed with UE loss and packet size. The algorithm was presented in 
    "Yu, Changsheng, et al. "Uplink scheduling and link adaptation for narrowband 
    Internet of Things systems." IEEE Access 5 (2017): 1724-1734."
    '''

    def __init__(self, dict):
        super().__init__(dict)
        self.past_actions = {}
        self.harq_ues = {}
        self.T = 20
        self.Imcs_dict = {(loss, size): 4 for loss in losses for size in p_bits}
        self.Irep_dict = {(loss, size): 0 for loss in losses for size in p_bits}
        self.error_count = {(loss, size): 0 for loss in losses for size in p_bits}
        self.sample_count = {(loss, size): 0 for loss in losses for size in p_bits}
        self.C_values = {(loss, size): 0 for loss in losses for size in p_bits}
        self.action_method = self.get_action_learning

    def action_required(self, sinr, info):
        if sinr > 0:
            return False
        ues = info.get('ues', [])
        if len(ues)==0:
            return False
        return True
    
    def deactivate_learning(self):
        self.action_method = self.get_action_plain
    
    def get_action(self, obs, r, info, action):
        '''executes the configured method'''
        return self.action_method(obs, r, info, action)
    
    def update_I_rep(self,l, p, counter):
        bler = self.error_count[(l,p)] / counter
        if bler < 0.07:
            r_ = self.Irep_dict[(l, p)]
            new_r = max(0, r_ - 1)
            self.Irep_dict[(l, p)] = new_r
        elif bler > 0.13:
            r_ = self.Irep_dict[(l, p)]
            new_r = min(MAX_I_REP - 1, r_ + 1)
            self.Irep_dict[(l, p)] = new_r

    def get_action_plain(self, obs, r, info, action):
        i = action[id_i] # gets the selected agent
        i = fix_ue_i(i, info)
        sinr = obs[sinr_i + user_vars * i]
        if not self.action_required(sinr, info): # in HARQ or when no UE
            if len(info['ues'])>0:
                ue_id = info['ues'][i]
                # self.harq_ues.append(ue_id)
                self.harq_ues.setdefault(ue_id, []).append(ue_id)
            return [0, 0] # dummy action
        else:
            loss = obs[loss_i + user_vars * i] # gets the loss of the selected UE
            buffer = obs[buffer_i + user_vars * i] # gets the buffer size of the selected UE
            ue_id = info['ues'][i]
            l_ = find_next_integer(losses, loss)
            b_ = find_next_integer(p_bits, buffer)
            I_mcs = self.Imcs_dict[(l_, b_)]
            I_rep = self.Irep_dict[(l_, b_)]
            conf = (I_mcs, I_rep)
        return conf

    def get_action_learning(self, obs, r, info, action):
        i = action[id_i] # gets the selected agent
        i = fix_ue_i(i, info)
        sinr = obs[sinr_i + user_vars * i]
        if not self.action_required(sinr, info): # in HARQ or when no UE
            if len(info['ues'])>0:
                ue_id = info['ues'][i]
                self.harq_ues.setdefault(ue_id, []).append(ue_id)
            return [0, 0] # dummy action
        else:
            loss = obs[loss_i + user_vars * i] # gets the loss of the selected UE
            buffer = obs[buffer_i + user_vars * i] # gets the buffer size of the selected UE
            ue_id = info['ues'][i]
            l_ = find_next_integer(losses, loss)
            b_ = find_next_integer(p_bits, buffer)
            I_mcs = self.Imcs_dict[(l_, b_)]
            I_rep = self.Irep_dict[(l_, b_)]
            conf = (I_mcs, I_rep)
            self.past_actions.setdefault(ue_id, []).append((l_, b_))

        # get the results of previous actions
        receptions = info['receptions']
        errors = info['errors']
        unfit = info['unfit']

        for ue_id in unfit:
            if ue_id in self.harq_ues:
                _ = get_element(self.harq_ues, ue_id) # simply remove it 
            else:
                _ = get_element(self.past_actions, ue_id) # simply remove it 
        
        # update the configurations 
        for ue_id in receptions.keys():
            if ue_id in self.harq_ues:
                _ = get_element(self.harq_ues, ue_id) # simply remove it 
                continue # do nothing else
            (loss, size) = get_element(self.past_actions, ue_id)
            # each reception (ACK) applies to smaller losses and smaller packets 
            # if configuration is equal (imcs and Nrep) or (-imcs and +Nrep)
            loss_values = [l for l in losses if l <= loss]
            packet_values = [p for p in p_bits if p <= size]
            imcs = self.Imcs_dict[(loss, size)]
            irep = self.Irep_dict[(loss, size)]
            for (l, p) in product(loss_values, packet_values):
                i_ = self.Imcs_dict[(l,p)]
                r_ = self.Irep_dict[(l,p)]
                if (i_ <= imcs) and (r_ >= irep):
                    self.sample_count[(l,p)] += 1
                    counter = self.sample_count[(l,p)]
                    if counter % self.T == 0: # update every T samples
                        self.update_I_rep(l, p, counter)
                    if i_ == 0: # L = L^{min}
                        if r_ == 0:
                            C = self.C_values[(l,p)]
                            new_C = min(C + 0.2, 5)
                            if new_C == 5:
                                self.Imcs_dict[(l,p)] += 1
                            self.C_values[(l,p)] = new_C
                        elif r_ > 0:
                            self.Irep_dict[(l,p)] -= 1
                    elif i_ == (I_MCS_MAX - 1): # L = L^{max}
                        if r_ > 0:
                            self.Irep_dict[(l,p)] -= 1
                    else:
                        C = self.C_values[(l,p)]
                        new_C = min(C + 0.2, 5)
                        if new_C == 5:
                            self.Imcs_dict[(l,p)] += 1
                        elif new_C == -5:
                            self.Imcs_dict[(l,p)] -= 1
                        self.C_values[(l,p)] = new_C
        
        for ue_id in errors:
            if ue_id in self.harq_ues:
                _ = get_element(self.harq_ues, ue_id) # simply remove it 
                continue
            (loss, size) = get_element(self.past_actions, ue_id) # extract and remove
            # each error (NACK) applies to larger losses and larger 
            # packets if configuration is equal or (+imcs and -Nrep)
            loss_values = [l for l in losses if l >= loss]
            packet_values = [p for p in p_bits if p >= size]
            imcs = self.Imcs_dict[(loss, size)]
            irep = self.Irep_dict[(loss, size)]
            for (l, p) in product(loss_values, packet_values):
                i_ = self.Imcs_dict[(l,p)]
                r_ = self.Irep_dict[(l,p)]
                if (i_ >= imcs) and (r_ <= irep):
                    self.sample_count[(l,p)] += 1
                    self.error_count[(l,p)] += 1
                    counter = self.sample_count[(l,p)]
                    if counter % self.T == 0: # update every T samples
                        self.update_I_rep(l, p, counter)
                    if i_ == 0: # L = L^{min}
                        if r_ < (MAX_I_REP - 1):
                            self.Irep_dict[(l,p)] += 1
                    elif i_ == (I_MCS_MAX - 1):
                        C = self.C_values[(l,p)]
                        # new_C = max(C - 1.8, -5)
                        new_C = max(C - 0.2, -5) # C_stepdown equal to C_stepup for better performance
                        if new_C == -5:
                            self.Imcs_dict[(l,p)] -= 1
                        self.C_values[(l,p)] = new_C
                    else:
                        C = self.C_values[(l,p)]
                        # new_C = max(C - 1.8, -5)
                        new_C = max(C - 0.2, -5) # C_stepdown equal to C_stepup for better performance
                        if new_C == 5:
                            self.Imcs_dict[(l,p)] += 1
                        elif new_C == -5:
                            self.Imcs_dict[(l,p)] -= 1
                        self.C_values[(l,p)] = new_C
        return conf

