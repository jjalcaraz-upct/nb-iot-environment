#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the classes needed to simulate one or several carriers
keeping track of the occupation of their resources (NPRACH and NPUSH)

Created on Jan 13, 2022

@author: juanjosealcaraz

"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from os import remove
from .utils import generate_movie
from .event_manager import schedule_event, Event
from .parameters import NPRACH_periodicity_list, NPRACH_N_sc_list, N_rep_preamble_list, Horizon, N_carriers

# resource configurations
resource_arrangements = [
    [0,1,2],
    [1,0,2],
    [0,2,1],
    [2,0,1],
    [1,2,0],
    [2,1,0]
]

# convert from preamble repetitions to subframes
NPRACH_N_sf_list = [int(np.ceil(N_rep * 5.6)) for N_rep in N_rep_preamble_list]

# conversion dictionaries
N_rep_to_N_sf = dict([(N_rep, int(np.ceil(N_rep * 5.6))) for N_rep in N_rep_preamble_list])
N_sf_to_N_rep = {value:key for key, value in N_rep_to_N_sf.items()} 

# useful dictionaries for UL grant insertion
offsets_per_sc = {
    1: [0,1,2,3,4,5,6,7,8,9,10,11],
    3: [0,3,6,9],
    6: [0,6],
    12: [0]
}

possible_delays = {
    8: [8, 16, 32, 64],
    16: [16, 32, 64],
    32: [32, 64],
    64: [64]
}

# for rendering and debug purposes
RACH_color = [0.2, 0.4, 0.6]
NPUSCH_color = [0.5, 0.6, 0.7, 0.8]
MSG3_colors = [0.2, 0.35, 0.5]

# CE_level: [periodicity, N_rep, N_sc (in 3.75 KHz) ]
default_conf_0 = [2, 0, 0]    # 160, 1, 12     
default_conf_1 = [2, 1, 0]    # 160, 2, 12 
default_conf_2 = [2, 2, 0]    # 160, 4, 12

# min lookahead for the carrier
MIN_SPAN = Horizon # observed subframes defined in parameters.py
N_carriers = N_carriers # number of subcarriers defined in parameters.py
# print(f' >> carrier.py: N_carriers = {N_carriers}')

# auxiliary function
def minpass(mymarks, mypass):
    passed = (x for x in mymarks if x >= mypass)
    min_value = next(passed) # this consumes the first value
    for x in passed: # this loops over all the remaining values
        if x < min_value:
            min_value = x
    return min_value

class CElevelSClog:
    '''
    auxiliary class to keep track of the total subcarriers allocated in
    non anchor carriers for each CE_level at every moment
    '''
    def __init__(self):
        self.log = {0: [0, 0, 0]}
    
    def register_sc(self, t, CE_level, sc):
        if t not in self.log.keys():
            self.log[t] = [0, 0, 0]
            self.log[t][CE_level] += sc

    def check_sc(self, t, CE_level):
        if t in self.log.keys():
            return self.log[t][CE_level] # how many subcarriers are avaiblable for this CE_level
        else:
            return 0
    
    def advance_time(self,current_t):
        for t in self.log.keys():
            if t < current_t:
                self.log.remove(t)


class NprachResource:
    '''
    Manages the parameters of a NPRACH resource for a CE_level, 
    and generates subframes according to these parameters.
    '''
    def __init__(self, periodicity, N_sf, N_sc, sc_offset = 0):
        self.t_start = 0
        self.t_next = self.t_start + periodicity
        self.sf_to_go = 0
        self.t_offset = 0
        self.normal_sf = np.full((12,1), False)
        self.conf_resource(periodicity, N_sf, N_sc, sc_offset)

    def sample(self, t):
        '''
        Provides two outputs:
        A boolean indicating if a new NPRACH resource has been scheduled
        An array of booleans indicating which subcarriers are used by the NPRACH
        '''
        if not self.N_sc: # the resource simply doesn't exist
            return False, self.normal_sf

        t -= self.t_offset

        if self.sf_to_go > 0:
            self.sf_to_go -= 1
            return False, self.NPRACH_sf
        elif t == self.t_next or not t:
            self.NPRACH_sf = self.create_new_prach_sf()
            self.t_start = t
            self.sf_to_go = self.N_sf - 1
            self.t_next = self.t_start + self.periodicity
            return True, self.NPRACH_sf
        else:
            return False, self.normal_sf

    def create_new_prach_sf(self):
        NPRACH_sf = np.full((12,1), False)
        NPRACH_sf[self.sc_offset: self.sc_offset + self.N_sc] = True
        return NPRACH_sf

    def conf_resource(self, periodicity, N_sf, N_sc, sc_offset):
        self.periodicity = periodicity
        self.N_sf = N_sf
        self.N_sc = N_sc
        self.sc_offset = sc_offset
    
    def update(self, ref_time, periodicity, N_sf, N_sc, sc_offset = 0, t_offset = 0):
        self.conf_resource(periodicity, N_sf, N_sc, sc_offset)
        self.t_next = self.t_start + periodicity
        if self.t_next < ref_time:
            self.t_next = ref_time + periodicity

        self.t_offset = t_offset

    def get_reference_time(self):
        return self.t_start

    def get_ra_parameters(self):
        N_rep = N_sf_to_N_rep[self.N_sf]
        N_sc = 4 * self.N_sc
        return N_rep, N_sc, self.periodicity


class SFGenerator:
    '''
    Manages the NPRACH parameters of one carrier for all CE_levels, 
    generates subframes according to these parameters and schedules 
    NPRACH events when required
    '''
    def __init__(self, NPRACH_list, log_sc_fn, carrier = 0):
        self.NPRACH_list = NPRACH_list
        self.carrier = carrier
        self.log_sc_fn = log_sc_fn
    
    def update(self, CE_level, periodicity, N_sf, N_sc, sc_offset = 0, t_offset = 0):
        reference_times = [r.get_reference_time() for r in self.NPRACH_list]
        rt = max(reference_times)
        self.NPRACH_list[CE_level].update(rt, periodicity, N_sf, N_sc, sc_offset = sc_offset, t_offset = t_offset)
    
    def next(self, t, color = False):
        '''
        generates a subframe, schedule NPRACH events, and registers the NPRACH
        resources that each CE_level has in each carrier 
        '''
        sf = np.full((12,1), False)
        for CE_level, NPRACH in enumerate(self.NPRACH_list):
            event, _sf = NPRACH.sample(t)
            if color:
                sf = sf + RACH_color[CE_level] * _sf
            else:
                sf = sf | _sf
            if event:
                N_rep, N_sc, _ = NPRACH.get_ra_parameters()
                if self.carrier == 0:
                    s = Event('NPRACH_start', CE_level = CE_level, carrier = self.carrier, N_sc = N_sc, N_rep = N_rep)
                    e = Event('NPRACH_end', CE_level = CE_level, carrier = self.carrier)
                    schedule_event(t, s)
                    schedule_event(t + NPRACH.N_sf, e)

                else:
                    self.log_sc_fn(t, CE_level, N_sc)
        return sf
    
    def get_conf(self):
        conf = {}
        for CE_level, NPRACH in enumerate(self.NPRACH_list):
            N_rep, N_sc, period = NPRACH.get_ra_parameters()
            conf[CE_level] = {'N_rep': N_rep, 'N_sc': N_sc, 'period': period}
        return conf


class Carrier:
    '''
    Manages the time-frequency resources of one or several carriers. 
    '''
    def __init__(self, m, n_carriers = N_carriers, animation = False, anim_step = 1, anim_span = 80):
        ''' creates a list of SFGenerators '''
        self.m = m
        self.m.set_carrier_set(self)
        self.n_carriers = n_carriers
        self.reset()
        
        if animation:
            self.activate_animation(anim_step = anim_step, anim_span = anim_span)
        else:
            self.animation = False

    def reset(self):
        self.sf_gen_list = []
        self.sf_ahead_list = []
        self.t_now = 0
        self.t_ahead = 0
        self.sc_log = CElevelSClog()
        for carrier_id in range(self.n_carriers):
            NPRACH_list = [
                NprachResource(*default_conf_0), 
                NprachResource(*default_conf_1),
                NprachResource(*default_conf_2)
            ]
            fn = self.sc_log.register_sc    
            self.sf_gen_list.append(SFGenerator(NPRACH_list, fn, carrier = carrier_id))
            self.sf_ahead_list.append(np.empty((12,0), dtype= np.bool))

    def activate_animation(self, anim_step = 1, anim_span = 80):
        self.animation = True            
        self.frame_paths = []
        self.frame_count = 0
        self.texts = {}
        self.anim_step = anim_step
        self.anim_span = anim_span

    def check_sc(self, t, CE_level):
        # provides subcarriers in non-anchor carriers
        if self.n_carriers == 1:
            return 0
        else:
            return self.sc_log.check_sc(t, CE_level)
    
    def get_state(self):
        # adds the states of all the carriers
        state = np.array([0.0]*MIN_SPAN, dtype=np.float_)
        for sf_ahead in self.sf_ahead_list:
            state += sf_ahead[:,0:MIN_SPAN].sum(axis=0)
        return 1.0 - state/(self.n_carriers * 12)

    def get_carrier_state(self, c):
        # returns the state of a specific carrier
        state = self.sf_ahead_list[c][:,0:MIN_SPAN].sum(axis=0)
        return 1.0 - state/12.0

    def extend_lookahead(self, new_t_ahead):
        # introduces new subframes (in all carriers)
        while self.t_ahead < new_t_ahead:
            for c, sf_gen in enumerate(self.sf_gen_list):
                self.sf_ahead_list[c] = np.hstack((self.sf_ahead_list[c], sf_gen.next(self.t_ahead, color = self.animation))) # +1 lookahead horizon
                # self.sf_ahead_list[c] = np.hstack((self.sf_ahead_list[c], sf_gen.next(self.t_ahead))) # +1 lookahead horizon
            self.t_ahead += 1

    def erase_past_sf(self, t):
        # removes past subframes (in all carriers)
        if self.animation:
            self.generate_frames(t)
        sf_past = max(t - self.t_now, 0)
        for c in range(self.n_carriers):
            self.sf_ahead_list[c] = np.delete(self.sf_ahead_list[c], np.s_[0:sf_past],1)
        self.t_now = max(t, self.t_now)
        self.extend_lookahead(t + MIN_SPAN)

    def generate_frames(self, t):
        t_slide = max(t - self.t_now, 0)
        t_span = self.anim_span
        t_step = self.anim_step
        # print(f'CARRIER BEFORE: t_ahead = {self.t_ahead}, t = {t}, t_now = {self.t_now}, t + t_span = {t + t_span}')
        # print(f'CARRIER BEFORE: sf_span length = {self.sf_ahead_list[0].shape[1]}')
        if self.t_ahead < t + t_span:
            self.extend_lookahead(t + t_span) 
        sf_span_list = self.sf_ahead_list
        # print(f'CARRIER AFTER: t_ahead = {self.t_ahead}, sf_span length = {sf_span.shape[1]}')
        for t_ in range(0, t_slide, t_step):
            if self.n_carriers > 1:
                fig, (ax, ax_) = plt.subplots(self.n_carriers, 1)
                axes = [ax, ax_]
            else:
                fig, ax = plt.subplots(1, 1)
                axes = [ax]
            for c_ in range(self.n_carriers):
                sf_span = sf_span_list[c_]
                shot = sf_span[:,t_:t_+t_span]
                # print(f' t_ = {t_}, t_span = {t_span}, shot length = {shot.shape[1]}')
                # masked_array = np.ma.masked_where(shot == 0, shot)
                # plt.imshow(masked_array, cmap = colormap, interpolation='nearest')
                ax = axes[c_]                  
                ax.imshow(shot, cmap = 'binary', interpolation='nearest', vmin=0, vmax=1)
                ax.set_xticks(np.arange(0, 80, 10))
                ax.set_xticks(np.arange(0, 80, 1), minor=True)
                ax.set_xticklabels(['{}'.format(x) for x in range(self.t_now+t_,self.t_now+t_+t_span,10)])
                ax.set_yticks(np.arange(0, 10, 5))
                ax.set_yticks(np.arange(0, 10, 2), minor=True)
                ax.grid(True)
            # insert texts:
            past_times = []
            for e_t, event in self.texts.items():
                text = event[0]
                offset = event[1]
                if e_t >= self.t_now and e_t <= self.t_now+t_+t_span:
                    ax.text(e_t - self.t_now - t_, offset, f'{e_t}: {text}', fontsize=9)
                if e_t < self.t_now:
                    past_times.append(e_t)
            for e_t in past_times:
                del self.texts[e_t]
            frame_path = f'frames/test_{10000000 + self.frame_count}.png'
            self.frame_paths.append(frame_path)
            fig.savefig(frame_path)
            plt.close()
            self.frame_count += 1

    def generate_movie(self, movie_name = 'movie'):
        if self.animation:
            generate_movie(movie_name = movie_name)
            for f_path in self.frame_paths:
                remove(f_path)
            self.frame_paths = []

    def get_conf(self):
        conf = []
        for carrier in range(self.n_carriers):
            conf.append(self.sf_gen_list[carrier].get_conf())
        return conf

    def step(self, t):
        # print(f'CARRIER * step * call to erase past t = {t}')
        self.erase_past_sf(t)
    
    def allocate_resources(self, carrier_id, t, delay = 8, N_sc = 1, N_sf = 1, CE_level = -1, UE_id = -1,  I_tbs = -1, N_rep = -1):
        '''
        allocates carrier resources for an UL grant
        carrier_id is readjusted if the allocation doesn't fit in the selected carrier
        delay can be readjusted as well
        if allocation is successful returns the last subframe occupied (sf) by the allocated resources
        '''   
        if self.animation: 
            if CE_level > -1:
                color =  MSG3_colors[min(CE_level, 2)]
            elif UE_id > -1:
                color = NPUSCH_color[UE_id % 4]
        else:
            color = True
        
        carrier_list = [carrier_id]
        carrier_list.extend([i for i in list(range(self.n_carriers)) if i != carrier_id])
        t_ul_end = 0
        offset = 0 # only for animation purposes
        time_in = 0 # only for animation purposes
        # print(f'CARRIER * allocate_resources * call to erase past t = {t}')
        self.erase_past_sf(t)

        for d in possible_delays[delay]:
            # print(f' CARRIER tries delay {d}')
            new_t_ahead = t + d + N_sf
            self.extend_lookahead(new_t_ahead)
            in_sf = slice(d, (d + N_sf))
            for (c, o) in product(carrier_list, offsets_per_sc[N_sc]):
                # print(f' CARRIER tries carrier {c} with sc offset {o}')
                in_sc = slice(o, (o + N_sc))
                sf_ahead = self.sf_ahead_list[c]
                if not sf_ahead[in_sc, in_sf].any():
                    t_ul_end = new_t_ahead
                    offset = o
                    time_in = t + d
                    sf_ahead[in_sc, in_sf] = color
                    # print(f' CARRIER fits NPUSCH in carrier {c} with sc offset {o} and delay {d}!')
                    break
            else:
                continue
            break

        if t_ul_end > 0 and self.animation:
            if CE_level > -1:
                self.texts[time_in] = (f'msg3 for CE {CE_level}', offset)
            elif UE_id > -1:
                self.texts[time_in] = (f'NPUSCH for UE {UE_id} ({I_tbs}, {N_rep})', offset)

        return t_ul_end

    def get_sf_ahead(self, c):
        return self.sf_ahead_list[c]

    def update_ra_parameters(self, CE_args_list):
        '''
        configures the parameters of all the NPRACH resources at each carrier
        receives a list with three lists inside
        each inner list contains [periodicity, N_sc (in 3.75 sc), N_reps]
        '''
        n_c = self.n_carriers
        periods = [NPRACH_periodicity_list[c_a[0]] for c_a in CE_args_list]
        N_sfs = [NPRACH_N_sf_list[c_a[1]] for c_a in CE_args_list]
        N_scs = [NPRACH_N_sc_list[n_c][c_a[2]] // 4 for c_a in CE_args_list]
        N_scs_copy = [NPRACH_N_sc_list[n_c][c_a[2]] // 4 for c_a in CE_args_list]
        N_scs_list = []
        for _ in range(self.n_carriers): # additional subcarriers correspond to non-anchor carriers
            N_scs_ = [min(N_sc, 12) for N_sc in N_scs]
            N_scs = [max(N_sc - N_sc_, 0) for N_sc, N_sc_ in zip(N_scs, N_scs_)]
            N_scs_list.append(N_scs_)

        periods, sc_offsets, sf_offsets = self.compute_offsets(periods, N_sfs, N_scs_list[0])

        for carrier in range(self.n_carriers):
            for ce in range(3):
                p = periods[ce]
                N_sf = N_sfs[ce]
                N_sc = N_scs_list[carrier][ce]
                sc_off = sc_offsets[ce]
                sf_off = sf_offsets[ce]
                self.sf_gen_list[carrier].update(ce, p, N_sf, N_sc, sc_offset = sc_off, t_offset = sf_off)
        
        return periods, N_scs_copy, N_sfs
    

    def compute_offsets(self, periods, N_sfs, N_scs):
        '''computes the offsets in subframes and in subcarriers for arranging 
        all the NPRACH resources without overlapping and also corrects the periodidicy
        if required'''
        sc_offsets = [0, 0, 0]
        sf_offsets = [0, N_sfs[0], N_sfs[0]+N_sfs[1]]
        min_length = sum(N_sfs)
        if sum(N_scs) > 12:
            for arrangement in resource_arrangements:
                i = arrangement[0]
                j = arrangement[1]
                k = arrangement[2]
                # the sum of subcarriers must be less than 12
                if N_scs[i] + N_scs[j] > 12:
                    continue
                # the one with the largest N_sf must go first
                if N_sfs[i] < N_sfs[j]:
                    continue
                # try the shorter configuration
                if N_scs[i] + N_scs[k] < 12:
                    if N_sfs[j] + N_sfs[k] < min_length:
                        min_length = N_sfs[j] + N_sfs[k]
                        sc_offsets[i] = 0
                        sc_offsets[j] = N_scs[i]
                        sc_offsets[k] = N_scs[i]
                        sf_offsets[i] = 0
                        sf_offsets[j] = 0
                        sf_offsets[k] = N_sfs[j]
                else: # try the other one
                    if N_sfs[i] + N_sfs[k] < min_length:
                        min_length = N_sfs[i] + N_sfs[k]
                        sc_offsets[i] = 0
                        sc_offsets[j] = N_scs[i]
                        sc_offsets[k] = 0
                        sf_offsets[i] = 0
                        sf_offsets[j] = 0
                        sf_offsets[k] = N_sfs[i]
        else:
            sc_offsets = [0, N_scs[0], N_scs[0]+N_scs[1]]
            sf_offsets = [0, 0, 0]
        
        min_periodicity = minpass(NPRACH_periodicity_list, min_length)

        periods = [max(min_periodicity, p) for p in periods]

        return periods, sc_offsets, sf_offsets
