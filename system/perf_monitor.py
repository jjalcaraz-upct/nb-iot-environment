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
from .utils import generate_movie
import copy

labels = ['Access delay', 'Transmission delay', 'Total delay']

class PerfMonitor:
    def __init__(self, m, reward_criteria = 'throughput', statistics = False, traces = False, animation = False, resource_monitoring = False):
        self.statistics = statistics
        self.animation = animation
        self.traces = traces
        self.resource_monitoring = resource_monitoring
        self.reward_criteria = reward_criteria
        m.set_perf_monitor(self)
        reward_functions = {
            'accumulated_delay': self.accumulated_delay,
            'average_delay': self.average_delay,
            'average_total_delay': self.average_total_delay,
            'throughput': self.throughput,
            'users': self.served_ues,
            'invd_users': self.invd_ues,
            'log_invd_users': self.log_invd_ues
        }
        self.reward_fn = reward_functions[reward_criteria]
        self.reset()
    
    def reset(self):
        self.t = 0
        self.error_count = 0
        self.attempts_count = 0
        self.thr_count = 0
        self.total_bits = 0
        self.ue_timestamp =  dict()
        self.ue_delays = dict()
        self.ue_unfit = []
        self.ue_errors = []
        self.ue_access_times = dict()
        self.ue_arrivals = [0,0,0]
        self.ue_connections = 0
        self.av_delay = 0
        self.ue_departures = [0,0,0]
        self.backoffs = [0,0,0]
        self.msg3_resources = 0 # subcarriers*subframes for msg3 signalling
        self.NPUSCH_resources = 0 # subcarriers*subframes for NPUSCHs
        self.NPRACH_resources = 0 # subcarriers*subframes for NPUSCHs

        if self.traces:
            self.rar_detections = [[],[],[]]
            self.rar_errors = [[],[],[]]
            self.rar_collisions = [[],[],[]]
            self.t_rar_detections = []
            self.nprach_arrivals = [[],[],[]]
            self.nprach_detections = [[],[],[]]
            self.t_nprach_arrivals = []
            self.t_nprach_detections = []

        if self.statistics or self.animation:
            self.npusch_delivered = [[],[],[]]
            self.RAR_in = [[], [], []]
            self.RAR_attmp = [[], [], []]
            self.RAR_sent = [[], [], []]
            self.RAR_detected = [[], [], []]
            self.msg3_resources_history = [] # subcarriers*subframes for msg3 signalling
            self.NPUSCH_resources_history = [] # subcarriers*subframes for NPUSCHs
            self.NPRACH_resources_history = [] # subcarriers*subframes for NPUSCHs
            self.access_history = []
            self.connection_history = []
            self.delay_history = []
            self.th_history = []
            self.histories = [self.access_history, self.delay_history, self.connection_history]
        
        if self.animation:
            self.frames = []
    
    def get_reward(self):
        return self.reward_fn()

    def get_info(self):
        info = {'receptions': self.ue_delays, 'errors': self.ue_errors, 'unfit': self.ue_unfit, 'access': self.ue_access_times}
        return info

    def clear_info(self):
        '''
        method called by the node b at each data scheduling step
        '''
        self.ue_delays =  dict()
        self.ue_access_times = dict()
        self.ue_unfit = []
        self.ue_errors = []

    def clear_nprach_metrics(self):
        '''
        method called by the node b at each nprach update step
        '''
        self.thr_count = 0

    def nprach_sample(self, n_ues, detections, CE, t):
        '''
        this is called from access procedure
        '''
        if self.traces:
            self.nprach_arrivals[CE].append((n_ues, t))
            self.nprach_detections[CE].append((detections, t))
            self.t_nprach_arrivals.append((n_ues, t))
            self.t_nprach_detections.append((detections, t))
    
    def rar_window_sample(self, CE_level, RAR_in, RAR_attmp, RAR_sent, RAR_ids):
        '''
        this is called from the node b to sample what happend in a RAR window
        '''
        if self.statistics:
            self.RAR_in[CE_level].append(RAR_in)
            self.RAR_attmp[CE_level].append(RAR_attmp)
            self.RAR_sent[CE_level].append(RAR_sent)
            self.RAR_detected[CE_level].append(RAR_ids)

    def rar_sample(self, detection, error, collisions, CE, t):
        '''
        this is called from rx_procedure to sample msg3 detections, errors, and collisions
        '''
        self.ue_connections += detection
        if self.traces:
            self.rar_detections[CE].append((detection, t))
            self.rar_errors[CE].append((error, t))
            self.rar_collisions[CE].append((collisions, t))
            self.t_rar_detections.append((detection, t))

    def arrival(self, CE_level):
        self.ue_arrivals[CE_level] += 1

    def backoff(self, CE_level):
        self.backoffs[CE_level] += 1

    def unregister_ue(self, ue):
        '''
        method used by the Node B to inform of a ue departure
        '''
        t_ = ue.t_disconnection
        delay = t_ - ue.t_connection
        service_time = t_ - ue.t_arrival # from arrival until the ue has finished tx
        ue_id = ue.id
        acked_data = ue.acked_data
        self.ue_departures[ue.CE_level] += 1
        self.thr_count += 1
        self.total_bits += acked_data
        self.av_delay += (delay - self.av_delay)/sum(self.ue_departures)
        if self.statistics or self.animation:  
            self.access_history.append(ue.t_connection - ue.t_arrival) # from arrival until ue is connected
            self.connection_history.append(service_time) # from arrival until the ue has finished tx
            self.delay_history.append(delay) # from ue connected until the ue has finished tx
            if len(self.npusch_delivered[ue.CE_level]) > 0 and self.npusch_delivered[ue.CE_level][-1][1] == t_:
                ues_ = self.npusch_delivered[ue.CE_level][-1][0]
                self.npusch_delivered[ue.CE_level][-1] = (ues_ + 1, t_)
            else:
                self.npusch_delivered[ue.CE_level].append((1, t_))
            ue_th = acked_data / delay
            self.th_history.append(ue_th)
            if self.animation:
                self.frames.append(copy.deepcopy(self.histories))
        del self.ue_timestamp[ue_id]
        return delay, service_time, acked_data

    def departure_ratio(self):
        return sum(self.ue_departures)/max(1,sum(self.ue_arrivals))
    
    def info_metrics(self, t):
        connection_rate = self.ue_connections/max(1,t)
        departure_rate = sum(self.ue_departures)/max(1,t)
        prod = departure_rate / max(1, self.av_delay)
        return connection_rate, departure_rate, self.av_delay, prod

    def register_msg3_resources(self, N_sc, N_sf):
        resources = N_sc * N_sf
        self.msg3_resources += resources
        if self.statistics:
            self.msg3_resources_history.append(resources)

    def register_NPUSCH_resources(self, N_sc, N_sf):
        resources = N_sc * N_sf
        self.NPUSCH_resources += resources
        if self.statistics:
            self.NPUSCH_resources_history.append(resources)

    def register_NPRACH_resources(self, N_sc, N_sf):
        resources = N_sc * N_sf
        self.NPRACH_resources += resources
        if self.statistics:
            self.NPRACH_resources_history.append(resources)

    def estimate_carrier_resources(self, t, n_carriers):
        '''
        estimation of the ratio of carrier resources occupied by RA signals
        and the ratio of free carrier space occupied by NPUSCH channels
        '''
        if t == 0:
            return 0.0, 0.0, 0.0
        period = t - self.t
        total_resources = 12 * n_carriers * period
        signalling_resources = self.NPRACH_resources + self.msg3_resources
        NPRACH_occupation = self.NPRACH_resources / total_resources
        RA_occupation = signalling_resources / total_resources
        no_RA_resources = total_resources - signalling_resources
        NPUSCH_occupation = self.NPUSCH_resources / no_RA_resources
        self.msg3_resources = 0
        self.NPRACH_resources = 0
        self.NPUSCH_resources = 0
        self.t = t
        return NPRACH_occupation, RA_occupation, NPUSCH_occupation

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
    
    def average_total_delay(self):
        '''
        average total delay (access + transmission) of connected users served in previous observation period
        '''
        tt = 0.0
        c = 0
        for ue_id, t_delay in self.ue_delays.items():
            a_delay = self.ue_access_times[ue_id]
            tt += t_delay + a_delay
            c += 1
        return -1.0 * tt/max(c,1)

    def throughput(self):
        '''
        served users during the last observation period
        '''
        return self.thr_count 

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