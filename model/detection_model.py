#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions that implement the detection model in NPRACH channels

The detection probability model is based on 
Vogt, Harald. "Efficient object identification with passive RFID tags." 
International Conference on Pervasive Computing. Berlin, Heidelberg: 
Springer Berlin Heidelberg, 2002.

Created on Dec 5, 2023

@author: juanjosealcaraz

"""

import pickle

from functools import lru_cache

with open('./model/p_detections_collisions.pickle', 'rb') as file:
    p_detections_collisions = pickle.load(file)

def ml_based_estimator(detections, collisions, N):
    '''
    maximum likelihood estimator of the number of contenders in an FSA frame with N opportunities
    given the observations, i.e. the number of idle opportunites, detections and collisions
    from "Efficient Random Access Channel Evaluation and Load Estimation in LTE-A With Massive MTC"
    '''
    n_opt = max(0, detections-1)
    p_max = p_detections_collisions.get((n_opt, detections, collisions, N),0.0)
    p_prev = p_max
    for n in range(n_opt,2*N+1):
        p = p_detections_collisions.get((n, detections, collisions, N), 0.0)
        if p > p_max:
            p_max = p
            n_opt = n
        if p < p_prev:
            return n_opt
        else:
            p_prev = p
    
    return n_opt

def estimate_incoming_traffic(d_list, c_list, N):
    '''
    computes the incoming traffic flow per window 
    d_list: list of detections
    c_list: list of collisions
    N: number of RA opportunities
    '''
    if not d_list:
        return 0
    estimations = [ml_based_estimator(d_,c_, N) for d_, c_ in zip(d_list, c_list)]
    return sum(estimations)/len(estimations)

##########################################################################################
# ################## Functions for computing conditional probabilities   #################
##########################################################################################

@lru_cache(maxsize=None)
def check_conditions(n, s, c, r):
    ''' 
    feasible combination of 
    n transmissions
    s detections
    c collisions
    r preambles
    '''
    if c < 0 or s < 0:
        return False

    # Check if s+c is less than or equal to r
    if not (s + c) <= r:
        return False
    
    # Check if there exists a beta such that s + βc = n, with β >= 2
    # Start with β = 2 and increment until β = N (upper bound since s, c >= 0)
    if not any(s + beta * c == n for beta in range(2, 2*r+1)):
        return False
    
    return True

@lru_cache(maxsize=None)
def P(n, s, c, r):
    '''
    conditional joint probability of having exactly s successes and c collisions, when n UEs transmitted their preambles 
    from "Efficient Random Access Channel Evaluation and Load Estimation in LTE-A With Massive MTC"
    n transmissions
    s detections
    c collisions
    r preambles
    '''
    # Base cases
    if n == 0 and s == 0 and c == 0:
        return 1
    if not check_conditions(n, s, c, r):
        return 0
    
    # Recursive calls
    term1 = (r - (s - 1 + c)) / r * P(n - 1, s - 1, c, r)
    term2 = (s + 1) / r * P(n - 1, s + 1, c - 1, r)
    term3 = c / r * P(n - 1, s, c, r)
    
    return term1 + term2 + term3


def compute_all_probabilities():
    p_detections_collisions = {}
    p_detections = {}
    counter = 0
    for r in [12, 24, 36, 48]:
        print(f'r: {r}')
        for n in range(2*r+1):
            for s in range(r+1):
                p_s_n_r = 0
                for c in range(r+1):
                    p_ = P(n, s, c, r)
                    if p_ > 0:
                        counter += 1
                        p_s_n_r += p_
                        p_detections_collisions[((n, s, c, r))] = p_
                p_detections[(s,n,r)] = p_s_n_r

    with open('./model/p_detections_collisions.pickle', 'wb') as file:
        pickle.dump(p_detections_collisions, file)

    with open('./model/p_detections.pickle', 'wb') as file:
        pickle.dump(p_detections, file)