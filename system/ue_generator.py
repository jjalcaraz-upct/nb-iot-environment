#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements two coexisting traffic models (uniform and beta) defined in 
"Study on RAN improvements for machine-type communications,” 
3GPP, Sophia Antipolis, France, Rep. TR 37.868, 2012.

Created on May 2022

@author: juanjosealcaraz

"""

from math import modf
from scipy import stats

T_beta = 10000 # 10 seconds in sf
T_uniform = 60000 # 60 seconds (1 minute) in sf

class UEGenerator:
    '''
    Generates the arrivals of UEs according to the configured traffic models
    '''
    def __init__(self, rng, m, M = 3000, alpha = 3, beta = 4, ratio = 1.0, buffer_range = [256, 256], random = False):
        self.rng = rng
        self.m = m
        self.m.set_ue_generator(self)
        self.ratio = ratio
        self.M = M
        self.dist = stats.beta(alpha, beta)
        self.buffer_range = buffer_range
        self.p_range = [0.0, 0.8]
        self.p = 0.5
        self.counter = 0
        self.random = random
        self.reset()

    def reset(self):
        self.M_uniform = round(self.M * min(self.ratio, 1.0))
        self.M_beta = self.M - self.M_uniform
        self.M_beta_max = self.M_beta
        self.t = 0

    
    def update_counter(self, t):
        new_count = t // T_uniform
        if new_count > self.counter:
            self.counter = new_count
            a = self.p_range[0]
            b = self.p_range[1]
            self.p = self.rng.uniform(a, b)
            # print(f' update counter {self.counter}, p = {self.p}')

    def generate_arrivals(self, t):
        if self.random:
            self.update_counter(t)
        t_elapsed = t - self.t
        arrivals = 0
        beta_arrivals = 0

        # first generate uniform arrivals
        if self.M_uniform > 0:
            intensity = self.M_uniform * (t_elapsed / T_uniform)
            (p, arrivals) = modf(intensity)
            arrivals += self.rng.binomial(1, p)
            self.M_uniform -= int(arrivals)

        # then add beta distributed traffic arrivals
        # if self.M_beta > 0 and self.t >= T_uniform:
        if self.M_beta > 0:
            t_s = self.t % T_uniform # start time
            t_s = t_s/T_beta # normalized
            t_e = t_s + t_elapsed/T_beta # elapsed time (normalized)
            intensity = self.M_beta * (self.dist.cdf(t_e) - self.dist.cdf(t_s))
            (p, beta_arrivals) = modf(intensity)
            beta_arrivals += self.rng.binomial(1, p)
            self.M_beta -= int(beta_arrivals)

        self.t = t

        return int(arrivals), int(beta_arrivals)

    def departure(self, ue):
        '''
        accounts for departures that are added to the population
        '''
        if self.random:
            if self.rng.binomial(1, self.p) > 0:
                self.M_beta += 1
            else:
                self.M_uniform += 1
        else:
            if ue.beta and self.M_beta < self.M_beta_max:
                self.M_beta += 1
            else:
                self.M_uniform += 1

    def generate_buffer(self):
        lower_bound = self.buffer_range[0]
        upper_bound = self.buffer_range[1]
        if upper_bound > lower_bound:
            return self.rng.integers(lower_bound, upper_bound+1)
        else:
            return lower_bound


if __name__=='__main__':
    from numpy.random import default_rng
    from .message_switch import MessageSwitch
    from .user import UE
    from .test_utils import moving_average
    import matplotlib.pyplot as plt
    rng = default_rng()
    m = MessageSwitch()
    generator = UEGenerator(rng, m, M = 2000, ratio = 0.8, buffer_range = [100, 500])
    print('-----------------------------------')
    print(f'M_uniform = {generator.M_uniform}')
    print(f'M_beta = {generator.M_beta}')
    print(f'M = {generator.M_beta + generator.M_uniform}')
    print('-----------------------------------')
    arrival_list = []
    beta_list = []
    total_list = []
    buffer_list = []
    for t in range(60000):
        if t % 50 == 0:
            arrivals, beta_arrivals = generator.generate_arrivals(t)
            arrival_list.append(arrivals)
            beta_list.append(beta_arrivals)
            total_list.append(arrivals + beta_arrivals)
            for _ in range(arrivals):
                buffer = generator.generate_buffer()
                buffer_list.append(buffer)
                ue = UE(t, buffer = buffer)
                generator.departure(ue)
            for _ in range(beta_arrivals):
                buffer = generator.generate_buffer()
                buffer_list.append(buffer)
                ue = UE(t, buffer = buffer)
                ue.beta = True
                generator.departure(ue)
        if t % 10000 == 0:
            print(f'M_beta = {generator.M_beta}')
    
    print('-----------------------------------')
    print(f'M_uniform = {generator.M_uniform}')
    print(f'M_beta = {generator.M_beta}')
    print(f'M = {generator.M_beta + generator.M_uniform}')
    print('-----------------------------------')
    print(f'total arrivals: {sum(total_list)}')
    print(f'uniform arrivals: {sum(arrival_list)}')
    print(f'beta arrivals: {sum(beta_list)}')
    print(f'average buffer: {sum(buffer_list)/len(buffer_list)}')
    fig1, ax1 = plt.subplots()
    ax1.plot(arrival_list, label = 'arrivals')
    ax1.plot(beta_arrivals, label = 'beta arrivals')
    ax1.plot(total_list, label = 'total arrivals')
    ax1.grid()
    ax1.legend(loc='best')
    fig1.savefig(f'arrival_process.png')
    
    buffer_history = moving_average(buffer_list, window = 500)
    fig2, ax2 = plt.subplots()
    ax2.plot(buffer_history)
    ax2.grid()
    fig2.savefig(f'buffer_history.png')