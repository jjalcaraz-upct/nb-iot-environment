#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary function for movie generation

Created on Jan 13, 2022

@author: juanjosealcaraz

"""

import subprocess
from pathlib import Path
import numpy as np
from scipy import stats
from .parameters import control_items, control_max_values, control_default_values, state_dim, N_carriers, Horizon

max_values = [v[N_carriers-1] for k,v in control_max_values.items() if k not in ['ce_level', 'rar_Imcs']] #  if k not in ['id', 'Imcs']
min_values = state_dim*[0]

action_basic = [v for k,v in control_default_values.items() if k not in ['ce_level', 'rar_Imcs']] # 

def generate_random_action(rng):
    return rng.integers(low = min_values, high = max_values)

carrier_i = control_items['carrier']
id_i = control_items['id']
sc_i = control_items['sc']

def generate_reasonable_action(rng, o):
    action = action_basic
    if N_carriers > 1:
        c0 = sum(o[range(state_dim - 2*Horizon, state_dim - Horizon)])
        c1 = sum(o[range(state_dim - Horizon, state_dim)])
        if c1 < c0:
            action[carrier_i] = 1
    action[id_i] = 0
    action[sc_i] = rng.integers(4) # subcarriers
    return action

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

# video creation
def ffmpeg_movie(frames_dir: Path, movie_dir: Path, framerate: int = 15) -> None:
    image_pattern = frames_dir / '*png'
    print(image_pattern)
    savepath = movie_dir.with_suffix('.mp4')
    print(savepath)

    # It's a lengthy command, but there's no need to grok it
    command = f"""
        ffmpeg -y -framerate {framerate} -f image2 -pattern_type glob \
        -i '{image_pattern}' -c:v libx264 -r 30 -profile:v high -crf 20 \
        -pix_fmt yuv420p {savepath}
    """
    subprocess.call(command, shell=True)

def generate_movie(movie_name = 'movie'):
    frames_dir = Path('./frames')
    movie_dir = Path(f'./movies/{movie_name}')
    ffmpeg_movie(frames_dir, movie_dir)

def find_next_integer(lst, n):
    '''
    Finds the smallest integer in the list 'lst' that is greater or equal than integer 'n'
    '''
    greater_integers = [i for i in lst if i >= n]
    if greater_integers:
        return min(greater_integers)
    else:
        return max(lst) # for safety
    
def find_next_integer_index(lst, n):
    """
    Finds the smallest integer in the list 'lst' that is greater or equal than integer 'n' and its position
    """
    greater_integers = [(i, j) for j, i in enumerate(lst) if i >= n]
    if greater_integers:
        return min(greater_integers)
    else:
        return max(lst), len(lst)-1