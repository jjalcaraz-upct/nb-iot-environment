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
    def __init__(self, rng, m, M = 3000, alpha = 3, beta = 4, ratio = 1.0, buffer_range = [256, 256]):
        self.rng = rng
        self.m = m
        self.m.set_ue_generator(self)
        self.ratio = ratio
        self.M = M
        self.dist = stats.beta(alpha, beta)
        self.buffer_range = buffer_range
        self.reset()

    def reset(self):
        self.M_uniform = round(self.M * min(self.ratio, 1.0))
        self.M_beta = self.M - self.M_uniform
        self.M_beta_max = self.M_beta
        self.t = 0

    def generate_arrivals(self, t):

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
