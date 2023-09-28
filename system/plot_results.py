#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for generating plots with the evolution of the delay

Created on Nov 11, 2022

@author: juanjosealcaraz

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from .utils import moving_average

'''
how to use this script:

python plot_results.py 1 1

'''

WINDOW = 200
RUNS = 30
Y_MAX = 20000
X_MAX = 15000
x = 4.5*1.2
y = 2.8*1.2

algo_list = ['A2C', 'PPO', 'DQN']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
types = ['1A', '2A']
y_axes_lim = {'1A': 22000, '2A': 20000}
x_axes_lim = {'1A': 18000, '2A': 16000}
mm_ticks = {'1A': (5000, 2500), '2A': (5000, 2500)}
scenarios = ['2000_10_B', '2000_10_BS']

args = sys.argv[1:]

# Parse the arguments
if len(args) >= 1:
    try:
        arg1 = int(args[0])
        type = types[arg1-1]
    except ValueError:
        print(f'Error: Argument 1 must be an integer between 1 and {len(types)}')
        sys.exit(1)
else:
    type = '1A'

if len(args) >= 2:
    try:
        arg2 = int(args[1])
        scenario = scenarios[arg2-1]
    except ValueError:
        print(f'Error: Argument 2 must be an integer between 1 and {len(scenarios)}')
        sys.exit(1)
else:
    scenario = '2000_10_B'

if len(args) >= 3:
    try:
        arg3 = int(args[2])
        Y_MAX = arg3
    except ValueError:
        print('Error: Argument 3 must be an integer')
        sys.exit(1)
else:
    Y_MAX = y_axes_lim[type]

if len(args) >= 4:
    try:
        arg4 = int(args[3])
        X_MAX = arg4
    except ValueError:
        print('Error: Argument 4 must be an integer')
        sys.exit(1)
else:
    X_MAX = x_axes_lim[type]


fig, ax = plt.subplots(figsize=(x,y), constrained_layout=True)

for i, algo in enumerate(algo_list):
    path = f'./results/{scenario}/{type}_{algo}/'
    histories = []
    n_users = []
    m_steps = np.inf
    runs = 0
    for filename in os.listdir(path):
        if filename.endswith(".npz"):
            hist = np.load(path + filename)
            h_ = hist['delay']
            runs += 1
            n_ = hist.get('served_users', [0])
            n_users.append(n_[0])
            delay_h = moving_average(h_, window = WINDOW)
            if len(delay_h) < m_steps:
                m_steps = len(delay_h)
            histories.append(delay_h)
    delays = histories.pop(0)[0:m_steps]
    for h in histories:
        delays = np.vstack((delays, h[0:m_steps]))

    delays_mean = np.mean(delays, axis=0)
    delays_std = np.std(delays, axis=0)

    steps = np.arange(len(delays_mean))

    n_users_mean = np.mean(n_users)
    n_users_std = np.std(n_users)

    # ax.set_title('Delay')
    ax.set_ylim((0,Y_MAX))
    ax.set_xlim((0,X_MAX))
    # ax.set_xlim((80000,100000))
    ax.plot(steps, delays_mean, label = algo, color = colors[i])
    ax.fill_between(steps, delays_mean - 1.697*delays_std/np.sqrt(runs),
                    delays_mean + 1.697*delays_std/np.sqrt(runs), color = '#DDDDDD')
    print(f'runs: {runs}, n_users_mean: {n_users_mean}')
    if n_users_mean > 0:
        ax.axvline(x=n_users_mean, linestyle='--', color = colors[i])
        ax.text(11000,11000,'DS phase')
ax.set_xlabel('NPUSCH transmissions')
ax.set_ylabel('Delay (ms)')
ax.legend(loc='best')
(M, m) = mm_ticks[type]
major_ticks = np.arange(0, Y_MAX+1, M)
minor_ticks = np.arange(0, Y_MAX+1, m)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# ax[i].grid()
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

fig.savefig(f'./figures/{type}_{scenario}.png', format='png', dpi = 600)
# fig.savefig(f'./figures/{type}_{scenario}_later.png', format='png')