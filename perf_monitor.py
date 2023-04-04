#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implements the PerfMonitor class which evaluates and keeps track of the system's performance.

Created on May 2022

@author: juanjosealcaraz

"""

import numpy as np
import matplotlib.pyplot as plt
from os import remove
from utils import generate_movie
import copy

labels = ['Access delay', 'Transmission delay', 'Total delay']

class PerfMonitor:
    def __init__(self, m, reward_criteria = 'invd_users', statistics = False, animation = False):
        self.statistics = statistics
        self.animation = animation
        m.set_perf_monitor(self)
        reward_functions = {
            'accumulated_delay': self.accumulated_delay,
            'average_delay': self.average_delay,
            'users': self.served_ues,
            'invd_users': self.invd_ues,
            'log_invd_users': self.log_invd_ues
        }
        self.reward_fn = reward_functions[reward_criteria]
        self.reset()
    
    def reset(self):   
        statistics = self.statistics
        animation = self.animation
        self.error_count = 0
        self.attempts_count = 0
        self.ue_timestamp =  dict()
        self.ue_delays = dict()
        self.ue_unfit = []
        self.ue_errors = []
        self.ue_access_times = dict()
        self.ue_arrivals = [0,0,0]
        self.ue_departures = [0,0,0]
        self.backoffs = [0,0,0]
        if statistics or animation:
            self.nprach_arrivals = [[],[],[]]
            self.nprach_detections = [[],[],[]]
            self.access_history = []
            self.connection_history = []
            self.delay_history = []
            self.histories = [self.access_history, self.delay_history, self.connection_history]
        if animation:
            self.frames = []


    def nprach_sample(self, n_ues, CE, arrival = False):
        if self.statistics:
            if arrival:
                self.nprach_arrivals[CE].append(n_ues)
            else:
                self.nprach_detections[CE].append(n_ues)

    def arrival(self, CE_level):
        self.ue_arrivals[CE_level] += 1

    def backoff(self, CE_level):
        self.backoffs[CE_level] += 1

    def unregister_ue(self, ue):
        '''
        method used by the Node B to inform of a ue departure
        '''
        t_ = ue.t_disconnection
        ue_id = ue.id
        self.ue_departures[ue.CE_level] += 1
        if self.statistics or self.animation:
            self.access_history.append(ue.t_connection - ue.t_arrival) # from arrival until ue is connected
            self.connection_history.append(t_ - ue.t_arrival) # from arrival until the ue has finished tx
            self.delay_history.append(t_ - ue.t_connection) # from ue connected until the ue has finished tx
            if self.animation:
                self.frames.append(copy.deepcopy(self.histories))
        del self.ue_timestamp[ue_id]

    def register_ue(self, ue):
        '''
        method used by the Node B to inform of a new ue connection
        '''
        self.ue_timestamp[ue.id] = ue.t_connection
        self.ue_access_times[ue.id] = ue.t_connection - ue.t_arrival

    def account_rx(self, t, ue):
        '''
        accounts for a successful packet reception while the ue is still connected
        '''
        self.ue_delays[ue.id] = t - self.ue_timestamp[ue.id]
        self.ue_timestamp[ue.id] = t
        self.attempts_count += 1
    
    def account_error(self, ue):
        '''
        accounts for an errored packet reception
        '''
        self.ue_errors.append(ue.id)
        self.error_count += 1
        self.attempts_count += 1

    def account_unfit(self, ue):
        '''
        accounts for configurations that did not fit
        '''
        self.ue_unfit.append(ue.id)

    def average_delay(self):
        '''
        average delay of connected users served in previous observation period
        '''
        tt = 0.0
        c = 0
        for t_delay in self.ue_delays.values():
            tt += t_delay
            c += 1
        return -1.0 * tt/max(c,1)

    def accumulated_delay(self):
        '''
        accumulated delay of users served in the observation period
        '''
        tt = 0.0
        for t_delay in self.ue_delays.values():
            tt += t_delay 
        return -1.0 * tt   

    def served_ues(self):
        '''
        total users served during the observation period
        '''
        return len(self.ue_delays)
    
    def total_departures(self):
        '''
        total users served up to date
        '''
        return sum(self.ue_departures)

    def get_error_count(self):
        return self.error_count, self.attempts_count 

    def invd_ues(self):
        '''
        sum of per-user rewards: each SERVED user provides a reward of 1/d where d is its delay in seconds
        '''
        inv_tt = 0.0
        for t_delay in self.ue_delays.values():
            inv_tt += 1.0/(t_delay / 1000.0) # delay is converted into seconds
        return inv_tt

    def log_invd_ues(self):
        '''
        sum of per-user rewards: each SERVED user provides a reward of -log(d) where d is its delay in seconds
        '''
        sum_log_tt = 0.0
        for t_delay in self.ue_delays.values():
            sum_log_tt += np.log(t_delay / 1000.0) # delay is converted into seconds
        return -1.0 * sum_log_tt

    def get_reward(self):
        return self.reward_fn()

    def get_info(self):
        info = {'receptions': self.ue_delays, 'errors': self.ue_errors, 'unfit': self.ue_unfit, 'access': self.ue_access_times}
        return info

    def clear_info(self):
        '''
        method called by the node b at each step
        '''
        self.ue_delays =  dict()
        self.ue_access_times = dict()
        self.ue_unfit = []
        self.ue_errors = []

    def generate_movie(self, movie_name = 'stats_movie'):
        if self.animation:
            frame_paths = []
            x_max = [max(h) for h in self.histories]
            x_min = [min(h) for h in self.histories]
            bins = [np.linspace(x_m, x_M, 20) for (x_m,x_M) in zip(x_min, x_max)]
            count = 0
            for frame in self.frames:
                fig, axes = plt.subplots(3,1)
                a = axes.ravel()
                for idx, ax in enumerate(a):
                    ax.grid(zorder=0, linestyle='--')
                    ax.hist(frame[idx], bins[idx], histtype='bar', zorder=3, color = "blue", lw=1)
                    ax.set_xlabel(labels[idx])
                fig.tight_layout()
                frame_path = f'frames/hist_{10000000 + count}.png'
                frame_paths.append(frame_path)
                fig.savefig(frame_path)
                plt.close()
                count += 1
            generate_movie(movie_name = movie_name)
            for f_path in frame_paths:
                remove(f_path)
            self.frames = []
            frame_paths = []
    
    def get_histories(self):
        return self.histories

    def get_throughput(self):
        return self.ue_arrivals, self.ue_departures, self.backoffs