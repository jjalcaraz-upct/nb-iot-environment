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
from system.utils import moving_average

'''
how to use this script:

python plot_multi_results.py 1 2
MAMBRL results for scenario 2000_10_BS

python plot_multi_results.py 2 1
NBLA results for scenario 2000_10_B
'''

WINDOW = 100
RUNS = 30
Y_MAX = 20000
X_MAX = 15000
x = 4.5*1.2
y = 2.8*1.2

algo_list = ['A2C', 'PPO', 'DQN']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
types = ['OL', 'NBLA']
x_axes_lim = {'OL': 6000, 'NBLA':30000}
y_axes_lim = {'OL': 250, 'NBLA':1200}
xy_text_label = {'OL': (3000, 150), 'NBLA': (15000, 900)}
mm_ticks = {'OL': (100, 50), 'NBLA': (400, 200)}
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
    type = 'OL'

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

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(x,y), constrained_layout=True, sharex=True)

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
    ax[i].set_ylim((0,Y_MAX))
    ax[i].set_xlim((0,X_MAX))
    ax[i].plot(steps, delays_mean, color = colors[i])
    ax[i].fill_between(steps, delays_mean - 1.697*delays_std/np.sqrt(runs),
                    delays_mean + 1.697*delays_std/np.sqrt(runs), color = '#DDDDDD')
    print(f'runs: {runs}, n_users_mean: {n_users_mean}')
    if n_users_mean > 0:
        ax[i].axvline(x=n_users_mean, linestyle='--', color = colors[i])
    ax[i].set_ylabel('Delay (ms)')
    (x_t, y_t) = xy_text_label[type]
    ax[i].text(x_t, y_t, f'DS phase with {algo}')

    (M, m) = mm_ticks[type]
    major_ticks = np.arange(0, Y_MAX+1, M)
    minor_ticks = np.arange(0, Y_MAX+1, m)

    ax[i].set_yticks(major_ticks)
    ax[i].set_yticks(minor_ticks, minor=True)

    # ax[i].grid()
    ax[i].grid(which='minor', alpha=0.2)
    ax[i].grid(which='major', alpha=0.5)

ax[i].set_xlabel('NPUSCH transmissions')

fig.savefig(f'./figures/{type}_{scenario}.png', format='png', dpi = 600)
# fig.savefig(f'./figures/{type}_{scenario}_later.png', format='png')