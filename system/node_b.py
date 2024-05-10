#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implements the Node B, which is the main entity of the system.
The node B is in charge, among other things, of advancind the discrete-event simulation process.

Created on Jan 13, 2022

@author: juanjosealcaraz

"""

from .event_manager import schedule_event, post_event, fel, Event
from .action_reader import ActionReader
from operator import attrgetter
import numpy as np
import pickle
from os import path
from . import parameters as par
from .user import STATE
from .utils import find_next_integer_index, find_next_integer

# DEBUG USE ONLY
tx_ues = []

# Node B states and processed events
NodeB_states = ['Scheduling', 'NPRACH_update', 'RAR_window', 'RAR_window_end','Idle']
NodeB_events = ['NPDCCH_arrival', 'NPRACH_update', 'RAR_window_end']

# action reader
reader = ActionReader()

# RAR msg_3 default configuration
min_th = par.threshold_list[0]
msg3_default_conf = [(2,1,1), (2,1,4), par.msg_3_conf[min_th]]
ce_lists = [[0,2,1],[1,2,0],[2,1,0]]

# CE level thresholds
th_values = par.th_values

# auxiliary functions
def average(lst):
    if len(lst) > 0:
        return round(sum(lst) / len(lst), 2)
    else:
        return 0
    
def itbs_bl(itbs, bl):
    ''' returns the block size closest to bl using itbs or the most similar one'''
    imcs = par.Itbs_to_Imcs[itbs]
    for i in range(imcs, 7):
        itbs = par.Imcs_to_Itbs[i]
        lst = par.tbs_lists[itbs]
        block_size, pos = find_next_integer_index(lst,bl)
        if block_size:
            return itbs, block_size, par.N_ru_list[pos]

def subtract_one(numbers):
    """
    Receives a list of integers and subtracts 1 from the first non-zero element in a more compact way.
    """
    # Finding the index of the first non-zero element
    for i, num in enumerate(numbers):
        if num > 0:
            numbers[i] -= 1
            break
    return numbers

def count_UEs(ue_lists):
    return [sum(r) for r in ue_lists]

class NodeB:
    '''
    Applies the actions at each step which corresponds either to a NPDCCH arrival or a NPRACH_update.
    It is instantiated with a message switch that connects it to the other entities.
    Handled events: NPDCCH_arrival, NPRACH_update
    Scheduled events: NPUSCH_end, RAR_window_end, NPRACH_update
    '''
    def __init__(self, m, sc_adjustment = False, mcs_automatic = False, ce_mcs_automatic = False, sort_criterium = 't_connection', tx_all_buffer = True):
        self.m = m # message switch
        self.m.set_node(self)

        # n carriers determined by the carrier set object
        self.n_carriers = self.m.carrier_set.n_carriers

        # NPDCCH parameters
        self.NPDCCH_sf_left = par.NPDCCH_sf

        # NPRACH parameters
        self.NPRACH_update_period = par.NPRACH_update_period
        self.NPRACH_conf = {}
        for item in par.NPRACH_items:
            self.NPRACH_conf[item] = par.control_default_values[item]

        # initialize storage structures
        self.reset_storage()

        # RAR parameters
        self.RAR_window_size = [4 * par.NPDCCH_period]*3 # can be between 2 and 10 (NPDCCH periods)
        self.valid_CE_control = True
        self.action = [0,0,0]
        self.final_action = [0,0,0] # check
        self.total_departures = 0

        # Node B parameters
        self.state = 'NPRACH_update'
        self.time = 0
        self.event = Event('NPRACH_update', t = 0)

        # optional configuration
        self.sc_adjustment = sc_adjustment
        self.mcs_automatic = mcs_automatic
        self.ce_mcs_automatic = ce_mcs_automatic
        self.sort_criterium = sort_criterium
        self.tx_all_buffer = tx_all_buffer
        if mcs_automatic:
            script_dir = path.dirname(path.abspath(__file__))
            data_file_path = path.join(script_dir, 'loss_tbs_to_mcs.p')
            self.loss_tbs_to_mcs = pickle.load(open(data_file_path, "rb"))
            # self.loss_tbs_to_mcs = pickle.load(open("./system/loss_tbs_to_mcs.p", "rb"))

        # step methods for each state
        self.step_methods = {
            'Scheduling': self.step_data,
            'RAR_window': self.step_RAR_window, 
            'RAR_window_end': self.step_RAR_end,
            'NPRACH_update': self.step_update,
        }

    def reset_storage(self):
        
        # RAR observations
        self.RAR_UEs = [[], [], []] # list with the numbers of UEs waiting for an msg2 at each CE level and at each RAR window
        self.RAR_w_ids = [[], [], []] # lists with active RAR windows
        self.RAR_in = [0, 0, 0] # list with the number of UEs detected in current NPRACH period per CE level
        self.RAR_ids = [0, 0, 0] # list with the number msg3 detected in current NPRACH period per CE level
        self.RAR_attmp = [0, 0, 0] # msg3 transmission attempts in current NPRACH period per CE level
        self.RAR_sent = [0, 0, 0] # msg3 transmissions in current NPRACH period per CE level
        self.RAR_detected = [0, 0, 0] # cumulative number of msg3 detections
        self.RAR_failed = [0, 0, 0] # msg3 that failed to be received in last closed window
        self.msg3_sent = [[], [], []] # msg3 sent in succesive NPRACH updates
        self.msg3_received = [[], [], []] # msg3 received in succesive NPRACH updates
        self.msg3_efficiency = [[], [], []] # msg3 detection ratios in succesive NPRACH updates
        # ------------------------------------------------------------

        # Connected UEs
        self.connected_UEs = dict() # all the backlogged ues
        self.selectable_UE_ids = [] # the ues on top of the queue

        # Scheduled UEs
        self.scheduled_UEs = dict()

        # connection / departure metrics for NPRACH configuration
        self.incoming = []
        self.departures = []
        self.service_times = []

        # distribution of the RSRP signal levels
        self.distribution = [0.0] * (len(th_values) + 1)
        self.k = 0

    def reset(self):
        # reset the future event list
        fel.reset()

        # tell the message broker to reset every entity
        self.m.reset()

        # reset storage structures
        self.reset_storage()

        # set initial state, time and event
        self.state = 'NPRACH_update'
        self.time = 0
        self.event = Event('NPRACH_update', t = 0)
        schedule_event(0, self.event)

        # schedule next event
        e = Event('NPDCCH_arrival')
        schedule_event(0, e)

        # put the simulation in motion
        self.advance_fel()

        # return initial state and info
        info = self.get_node_info()

        return info

    def advance_fel(self):
        '''
        extracts events from the fel until reaching a state where an action is required
        '''
        while True:
            t, event = fel.pop_next()
            self.time = t
            self.m.carrier_set.step(t)
            if event.type in ['NPDCCH_arrival', 'NPRACH_update', 'RAR_window_end']:
                self.event = event
                if self.check_event(event.type): # checks if a control action is required
                    break
            else:
                post_event(event)


    def distribution_update(self, sample):
        '''
        updates the estimation of the RSRP signal power of the users
        '''
        if self.k > 1000: # maximum number of samples to estimate the power distribution
            return
        
        self.k += 1   
        # Determine the interval the sample falls into
        interval_index = next((i for i, threshold in enumerate(th_values) if sample + 35 < threshold), len(th_values))

        # Update probabilities
        for i, prob in enumerate(self.distribution):
            if i == interval_index:
                # Sample falls in this interval
                self.distribution[i] += (1 - prob) / self.k
            else:
                # Sample does not fall in this interval
                self.distribution[i] += (0 - prob) / self.k


    def step(self, action):
        ''' 
        Applies the action and then advances the FEL until the next handled event
        '''
        # run the appropriate function given the state (scheduling , RAR_window, RAR_window_end, NPRACH_update)
        step_fn = self.step_methods[self.state]
        step_fn(action)

        # advance forward until next control event
        self.advance_fel()

        # get general state
        info = self.get_node_info()

        # get reward
        reward = self.m.perf_monitor.get_reward()

        return reward, False, info


    def check_event(self, type):
        '''
        checks if a control action is required
        '''
        brk_cond = False
        if type == 'NPDCCH_arrival':
            if self.state in ['RAR_window', 'Scheduling']:
                brk_cond = True
            else:
                self.schedule_NPDCCH_arrival(self.NPDCCH_sf_left)
        elif type in ['NPRACH_update', 'RAR_window_end']:
            self.state = type
            brk_cond = True
        return brk_cond


    def update_state(self):
        '''
        determines the state of the node
        '''
        RAR_ues = count_UEs(self.RAR_UEs)
        # RAR_ids = count_UEs(self.RAR_w_ids)
        # if sum(RAR_ids) and sum(RAR_ues):
        if sum(RAR_ues):
            self.state = 'RAR_window'
        elif len(self.connected_UEs) > 0:
            self.state = 'Scheduling'
        else:
            self.state = 'Idle'
         
    def get_node_info(self):
        '''
        extracts the full info from the node:
        in Scheduling activates:
        info of up to N_users (4) (loss, newData, connection_time) + total users

        in RAR_window activates
        number of detections per CE_level 

        both in Scheduling and RAR_window:
        carrier_state 

        in NPRACH update activates
        per CE_level:
        - periodicity
        - subcarriers
        - repetitions
        - detections per NPRACH during last period
        '''
        total_ues = len(self.connected_UEs)
        carrier_state = self.m.carrier_set.get_carrier_state(0)
        info = {'time': self.time, 'state': self.state, 'total_ues': total_ues, 'carrier_state': carrier_state}
        
        if self.state in ['Scheduling','RAR_window_end']: # no RAR window
            selectable_ues = []
            sf_left = self.NPDCCH_sf_left
            max_CE = par.CE_for_NPDCCH_sf_left[sf_left]
            n_ue = 0       
            for ue in sorted(self.connected_UEs.values(), key=attrgetter(self.sort_criterium)):
                if ue.CE_level <= max_CE:
                    selectable_ues.append(ue)
                    n_ue += 1
                if n_ue == par.N_users: # N_users in parameters.py: the n users observed  
                    break
            self.selectable_UE_ids = [ue.id for ue in selectable_ues] # selectable ues until next decision
            info['ues'] = self.selectable_UE_ids

            connection_time = [0.0]*par.N_users
            loss = [0.0]*par.N_users
            sinr = [0.0]*par.N_users
            buffer = [0.0]*par.N_users

            for i, ue in enumerate(selectable_ues):
                connection_time[i] = self.time - ue.t_connection
                loss[i] = ue.loss
                buffer[i] = ue.buffer
                if not ue.new_data:
                    sinr[i] = min(max(ue.sinr, -40),30) + 40 # [0, 70]
            
            info['connection_time'] = connection_time
            info['loss'] = loss
            info['sinr'] = sinr
            info['buffer'] = buffer

            perf_info = self.m.perf_monitor.get_info()
            info.update(perf_info)

        elif self.state == 'RAR_window': # RAR window
            sf_left = self.NPDCCH_sf_left
            max_CE = par.CE_for_NPDCCH_sf_left[sf_left]
            RAR_ues = count_UEs(self.RAR_UEs)
            RAR_ues = [ue if i<= max_CE else 0 for i,ue in enumerate(RAR_ues)]
            
            info['ues_per_CE'] = RAR_ues
            info['RAR_in'] = self.RAR_in
            info['RAR_sent'] = self.RAR_sent
            info['RAR_ids'] = self.RAR_ids
            info['RAR_detected'] = self.RAR_detected
            info['RAR_failed'] = self.RAR_failed
            info['valid_CE_control'] = self.valid_CE_control
            info['NPDCCH_sf_left'] = self.NPDCCH_sf_left
            info['carrier_state'] = self.m.carrier_set.get_carrier_state(0)
            info['total_departures'] = self.total_departures
            info['sc_C0'] = self.NPRACH_conf['sc_C0']
            info['sc_C1'] = self.NPRACH_conf['sc_C1']
            info['sc_C2'] = self.NPRACH_conf['sc_C2']
            info['period_C0'] = self.NPRACH_conf['period_C0']
            info['period_C1'] = self.NPRACH_conf['period_C1']
            info['period_C2'] = self.NPRACH_conf['period_C2']
            info['th_C1'] = self.NPRACH_conf['th_C1']
            info['th_C0'] = self.NPRACH_conf['th_C0']
            info['action'] = self.action
            info['final_action'] = self.action

            self.RAR_failed = [0, 0, 0]
            
        else: # NPRACH update
            num_departures = len(self.departures)
            detection_h, colision_h = self.m.access_procedure.get_histories()
            NPRACH_occupation, RA_occupation, NPUSCH_occupation = self.m.perf_monitor.estimate_carrier_resources(self.time, self.n_carriers)
            av_delay = sum(self.departures)/num_departures if num_departures else 0
            p_0 = (self.NPRACH_conf['sc_C0'] + 1) * 12
            p_1 = (self.NPRACH_conf['sc_C1'] + 1) * 12
            p_2 = (self.NPRACH_conf['sc_C2'] + 1) * 12
            preambles = [p_0, p_1, p_2]
            msg3_detec_ratio = [0.0, 0.0, 0.0]
            msg3_avg_sent = [0.0, 0.0, 0.0]
            msg3_avg_received = [0.0, 0.0, 0.0]
            detection_ratios = [0.0, 0.0, 0.0]
            colision_ratios = [0.0, 0.0, 0.0]
            for ce in range(3):
                d_r = self.msg3_efficiency[ce]
                s_r = self.msg3_sent[ce]
                r_r = self.msg3_received[ce]
                d_h_ = detection_h[ce]
                c_h_ = colision_h[ce]
                msg3_detec_ratio[ce] = sum(d_r)/max(1,len(d_r))
                msg3_avg_sent[ce] = sum(s_r)/max(1,len(s_r))
                msg3_avg_received[ce] = sum(r_r)/max(1,len(r_r))
                detection_ratios[ce] = sum(d_h_)/max(1,len(d_h_))/preambles[ce]
                colision_ratios[ce] = sum(c_h_)/max(1,len(c_h_))/preambles[ce]

            # info
            info['detection_ratios'] = detection_ratios # *
            info['colision_ratios'] = colision_ratios # *
            info['distribution'] = self.distribution # *
            info['msg3_detection'] = msg3_detec_ratio # *
            info['msg3_sent'] = msg3_avg_sent
            info['msg3_received'] = msg3_avg_received
            info['preambles'] = preambles
            info['NPRACH_detection'] = detection_h
            info['NPRACH_collision'] = colision_h
            info['RA_occupation'] = RA_occupation 
            info['NPUSCH_occupation'] = NPUSCH_occupation # *
            info['NPRACH_occupation'] = NPRACH_occupation # *
            info['incoming'] = self.incoming
            info['departures'] = num_departures
            info['delays'] = self.departures
            info['av_delay'] = av_delay # *
            info['service_times'] = self.service_times
            info['NPRACH conf'] = self.NPRACH_conf

            self.msg3_efficiency = [[],[],[]] # reset
            self.msg3_sent = [[],[],[]] # reset
            self.msg3_received = [[],[],[]] # reset
            self.departures = []
            self.service_times = []
            self.incoming = []
            self.m.access_procedure.reset_history()

        return info

    def schedule_NPDCCH_arrival(self, elapsed_sfs):
        '''
        Auxiliary method for re-scheduling a NPDCCH_arrival event after processing one
        '''
        self.NPDCCH_sf_left = max(self.NPDCCH_sf_left - elapsed_sfs, 0)

        # re-schedule NPDCCH_arrival
        next_time = self.time + elapsed_sfs
        if not self.NPDCCH_sf_left:
            next_time += par.NPDCCH_period
            self.NPDCCH_sf_left = par.NPDCCH_sf

        schedule_event(next_time, self.event)


    def allocate_resources(self, c_id, ref_t, delay, N_sc, N_ru, N_rep, CE_level = -1, UE_id = -1, I_tbs = -1):
        '''
        allocate resources in the selected carrier with a given delay and returns the end-of-tx time
        if the resources do not fit with the selected number of subcarriers it retries with a different number
        '''
        n_sc = N_sc
        N_sf = par.N_sf_per_ru[N_sc] * N_ru * N_rep
        t_ul_end = self.m.carrier_set.allocate_resources(c_id, ref_t, delay = delay, N_sc = N_sc, N_sf = N_sf, CE_level = CE_level, UE_id = UE_id, I_tbs = I_tbs, N_rep = N_rep)
        # NB tries to allocate resources with N_sc = {N_sc}, N_sf = {N_sf}'
        if not t_ul_end and self.sc_adjustment: # automatic adjustment of subcarriers
            for n_sc in par.N_sc_selection_list[N_sc]:
                N_sf = par.N_sf_per_ru[n_sc] * N_ru * N_rep
                t_ul_end = self.m.carrier_set.allocate_resources(c_id, ref_t, delay = delay, N_sc = n_sc, N_sf = N_sf, CE_level = CE_level, UE_id = UE_id, I_tbs = I_tbs, N_rep = N_rep)
                if t_ul_end:
                    return t_ul_end, n_sc, N_sf
        return t_ul_end, n_sc, N_sf


    def step_RAR_window(self, action):
        '''
        Schedules an UL grant for the msg3 of one user
        the action determines which CE level should receive the msg
        and the tx parameters of the msg3
        '''
        # extract the parameters from the action
        RAR_params = reader.scheduling_action(action, RAR_action = True)
        CE_level = RAR_params['i']
        I_tbs = RAR_params['I_tbs']
        N_ru = RAR_params['N_ru']
        N_rep = RAR_params['N_rep']
        self.action = [CE_level, I_tbs, N_rep]

        RAR_ues = count_UEs(self.RAR_UEs)

        self.valid_CE_control = True
        valid_control = True        

        NPDCCH_sf = par.NPDCCH_sf_per_CE[CE_level] # DCI subframes depend on CE level

        if not RAR_ues[CE_level] > 0 or NPDCCH_sf > self.NPDCCH_sf_left or self.ce_mcs_automatic:
            # if the selected CE_level is not valid or if the ce selection is automatic    
            sorted_CEs = [index for index, _ in sorted(enumerate(self.RAR_UEs), key=lambda pair: pair[1], reverse=True)]
            # check if this control is useful and, if not, try with the others        
            valid_control = False
            # for CE_level in ce_lists[ce]:
            for CE_level in sorted_CEs:
                NPDCCH_sf = par.NPDCCH_sf_per_CE[CE_level] # required NPDCCH repetitions
                msgs_to_send = RAR_ues[CE_level] > 0
                enough_signalling_space = self.NPDCCH_sf_left >= NPDCCH_sf
                valid_control = msgs_to_send and enough_signalling_space
                if valid_control:
                    self.valid_CE_control = False # the selected CE was not valid
                    break

        # no valid control found
        if not valid_control:
            self.schedule_NPDCCH_arrival(self.NPDCCH_sf_left)        
            return

        # extract every required parameter
        carrier_id = RAR_params['carrier']
        delay = RAR_params['delay']
        N_sc = RAR_params['N_sc']

        # if self.ce_mcs_automatic or not self.valid_CE_control: 
        if self.ce_mcs_automatic: # for automatic configuration we use predefined I_tbs N_rep and N_ru
            (I_tbs, N_ru, N_rep) = msg3_default_conf[CE_level]
        
        self.final_action = [CE_level, I_tbs, N_rep]

        # check if the allocated resources fit into the selected carrier
        reference_time = self.time + NPDCCH_sf
        t_ul_end, N_sc, n_sf = self.allocate_resources(carrier_id, reference_time, delay, N_sc, N_ru, N_rep, CE_level = CE_level)
        # if the resources fit, submit an UL grant for msg3
        self.RAR_attmp[CE_level] += 1
        
        if t_ul_end:
            self.m.population.msg3_grant(CE_level, reference_time, t_ul_end, I_tbs = I_tbs, N_rep = N_rep)
            self.RAR_sent[CE_level] += 1
            self.RAR_UEs[CE_level] = subtract_one(self.RAR_UEs[CE_level])
            self.m.perf_monitor.register_msg3_resources(N_sc, n_sf)
        else:
            self.valid_CE_control = False

        # re-schedule NPDCCH
        self.schedule_NPDCCH_arrival(NPDCCH_sf)
        self.update_state()


    def start_RAR_window(self, detections, CE_level, time, rar_w_id):
        ''' 
        Method invoked by AccessProcedure to notify the start of the RAR window,
        number_of_UEs: number of detected preambles
        CE_level: CE level of the RAR window
        '''
        # update RAR state
        self.RAR_UEs[CE_level].append(detections)
        self.RAR_w_ids[CE_level].append(rar_w_id) # active windows
        self.state = 'RAR_window'

        # RAR metrics from previous window
        RAR_in = self.RAR_in[CE_level]
        RAR_detections = self.RAR_ids[CE_level]
        RAR_attmp = self.RAR_attmp[CE_level]
        RAR_sent = self.RAR_sent[CE_level]
        self.msg3_sent[CE_level].append(RAR_sent) # info
        self.msg3_received[CE_level].append(RAR_detections) # info
        self.msg3_efficiency[CE_level].append(RAR_detections / max(1,RAR_in)) # insert msg3 detection ratio
        self.m.perf_monitor.rar_window_sample(CE_level, RAR_in, RAR_attmp, RAR_sent, RAR_detections)

        # reset RAR metrics
        self.RAR_in[CE_level] = detections
        self.RAR_ids[CE_level] = 0 # list with the number msg3 detected in current RAR window at each CE level
        self.RAR_attmp[CE_level] = 0 # attempts
        self.RAR_sent[CE_level] = 0
        
        # schedule end of this RAR window for this CE_level
        e_time = time + self.RAR_window_size[CE_level]
        RAR_window_end = Event('RAR_window_end', t = e_time, CE_level = CE_level, rar_w_id = rar_w_id)
        schedule_event(e_time, RAR_window_end)

        ################################################

    def step_RAR_end(self, action):
        '''
        Processes a 'RAR_window_end' event associated to a given CE_level.
        Communicates the backoff value to the UEs
        '''
        CE_level = self.event.CE_level
        rar_w_id = self.event.rar_w_id
        t = self.event.t

        backoff = reader.UE_contention_resolution(action)
        period_i = self.NPRACH_conf[f'period_C{CE_level}']
        backoff = max(backoff, par.backoff_list[period_i + 1])

        # report users
        _ = self.m.population.RAR_window_end(t, CE_level, backoff, rar_w_id)
        
        if self.RAR_UEs[CE_level]: # just in case
            self.RAR_failed[CE_level] = self.RAR_UEs[CE_level].pop(0)

        if self.RAR_w_ids[CE_level]:
            self.RAR_w_ids[CE_level].pop(0)

        # update state
        self.update_state()


    def step_data(self, action):
        '''
        Allocates an UL grant for the data transmission of one connected UE
        '''
        # we start a new epoch of performance sampling
        self.m.perf_monitor.clear_info()

        # list of ues
        ue_id_list = self.selectable_UE_ids
        if len(ue_id_list) == 0:
            # nothing to do here, no selectable UEs: re-schedule NPDCCH
            self.schedule_NPDCCH_arrival(self.NPDCCH_sf_left)
            self.update_state()
            return

        # extract the parameters from the action
        data_params = reader.scheduling_action(action)

        # extract the UE_id
        ue_i = data_params['i']
        if ue_i >= len(ue_id_list):
            ue_i = 0 # the oldest ue in the list

        ue_id = ue_id_list[ue_i]
        ue = self.connected_UEs[ue_id]
        CE_level = ue.CE_level
        new_data = ue.new_data

        if not new_data:
            I_tbs = ue.I_tbs # for HARQ we will use the same MCS
            N_rep = ue.N_rep # and the same number of repetitions
            N_ru = ue.N_ru
            tbs = ue.tbs

        elif self.mcs_automatic: # for automatic configuration we use predefined I_tbs N_rep and N_ru
            loss = round(ue.loss, 1)
            loss = max(121.4, min(171.4, loss))
            tbs = find_next_integer(par.TBS_list, ue.buffer) # assume all the buffer is sent
            (I_tbs, N_rep, N_ru) = self.loss_tbs_to_mcs[loss, tbs]
        
        else: # mcs comes from the agent
            N_rep = data_params['N_rep']
            I_tbs = data_params['I_tbs']
            # now determine N_ru and tbs
            if self.tx_all_buffer:
                # the N_ru, tbs pair is determined automatically from the I_tbs and the buffer size
                I_tbs, tbs, N_ru = itbs_bl(I_tbs, ue.buffer) # upgrades the I_tbs if required to fit the buffer and gets the block size
            else:
                # the N_ru and thus the tbs is determined by the agent
                lst = par.tbs_lists[I_tbs]
                tbs, pos = find_next_integer_index(lst, ue.buffer)
                I_N_ru = data_params['N_ru'] # this is an index in the tbs_list
                if pos < I_N_ru:
                    N_ru = par.N_ru_list[pos]
                else:
                    N_ru = par.N_ru_list[I_N_ru]
                    tbs = lst[I_N_ru]

        # extract other parameters
        carrier_id = data_params['carrier']
        delay = data_params['delay']
        N_sc = data_params['N_sc']

        # get the required number of NPDCCH repetitions
        NPDCCH_sf = par.NPDCCH_sf_per_CE[CE_level]

        # check if the allocated resources fit into the selected carrier
        reference_time = self.time + NPDCCH_sf
        t_ul_end, N_sc, n_sf = self.allocate_resources(carrier_id, reference_time, delay, N_sc, N_ru, N_rep, UE_id = ue_id, I_tbs = I_tbs)
        # if the resources fit, submit an UL grant for PUSCH
        if t_ul_end:
            # NB NPUSCH grant allocated with ref time {reference_time} and t_ul_end = {t_ul_end}
            self.data_grant(ue_id, t_ul_end, I_tbs = I_tbs, N_rep = N_rep, N_ru = N_ru, tbs = tbs)
            self.m.perf_monitor.register_NPUSCH_resources(N_sc, n_sf) 
        elif new_data:
            #  NB NPUSCH with N_sc = {N_sc} I_tbs={I_tbs}, N_rep={N_rep} for ue {ue.id} did not fit
            self.m.perf_monitor.account_unfit(ue)
        
        # re-schedule NPDCCH
        self.schedule_NPDCCH_arrival(NPDCCH_sf)
        self.update_state()


    def data_grant(self, UE_id, t_end, I_tbs = 1, N_rep = 2, N_ru = 1, tbs = 256):
        '''
        method to assign an UL transmission grant to a specific UE
        '''
        ue = self.connected_UEs.pop(UE_id) # extract the ue 
        
        ue.I_tbs = I_tbs
        ue.N_rep = N_rep
        ue.N_ru = N_ru
        ue.tbs = tbs # used by the channel to compute the error probability

        if ue.retx_buffer > 0: # only one HARQ process
            ue.bits = ue.retx_buffer

        elif tbs >= ue.buffer: # transmit all the buffered bits
            ue.bits = ue.buffer
            ue.buffer = 0

        else:
            ue.bits = min(ue.buffer, tbs)
            ue.buffer -= ue.bits
        
        # NB t={self.time} assigns UL grant to ue {UE_id} to transmit {ue.bits} bits with sinr {ue.sinr}. STATE = {self.state}
        self.scheduled_UEs[ue.id] = ue
        
        e = Event('NPUSCH_end', ue = ue) # schedule NPUSCH_end event (handled by RxProcedure)
        schedule_event(t_end, e)

    def step_update(self, action):
        '''
        Updates the NPRACH parameters
        '''
        # we start a new epoch of performance sampling
        self.m.perf_monitor.clear_nprach_metrics()

        # extract parameters
        NPRACH_conf = reader.NPRACH_update_action(action)

        # configure RSRP CE levels
        th_C1 = par.threshold_list[NPRACH_conf['th_C1']]
        th_C0 = par.threshold_list[NPRACH_conf['th_C0']]

        if th_C1 >= th_C0:
            th_C0 = 0 # use only two levels: CE1 and CE2

        # adjust n_rep to CE level
        min_th = par.threshold_list[0]
        rep_C2 = par.thr_to_n_rep[min_th]
        if th_C1 < par.no_rep_th:
            th_C1 = max(min_th, th_C1)
            rep_C1 = par.thr_to_n_rep[th_C1]
        else:
            rep_C1 = 0
        if th_C0 < par.no_rep_th:
            th_C0 = max(min_th, th_C0)
            rep_C0 = par.thr_to_n_rep[th_C0]
        else:
            rep_C0 = 0

        # adjust msg3_conf to CE level
        if th_C1 < par.no_rep_msg3_th:
            msg3_default_conf[1] = par.msg_3_conf[th_C1]
        else:
            msg3_default_conf[1] = (2,1,1)

        if th_C0 < par.no_rep_msg3_th:
            msg3_default_conf[0] = par.msg_3_conf[th_C0]
        else:
            msg3_default_conf[0] = (2,1,1)
              
        # obtain RAR_window_size
        RAR_window_size = par.RAR_WindowSize_list[NPRACH_conf['rar_window']] * par.NPDCCH_period
        
        # inform the UEs
        UE_kwargs = {
            'MAC_timer': par.MAC_timer_list[NPRACH_conf['mac_timer']] * par.NPDCCH_period, 
            'preamble_trans_max_CE': par.preamble_trans_max_CE_list[NPRACH_conf['transmax']],
            'probability_anchor': par.probability_anchor[NPRACH_conf['panchor']]
            }
        self.m.population.update_CE_thresholds(th_C1, th_C0)
        self.m.population.update_ra_parameters(**UE_kwargs)

        # configure carrier
        CE_args_list = [[NPRACH_conf['period_C0'], rep_C0, NPRACH_conf['sc_C0']],
                [NPRACH_conf['period_C1'], rep_C1, NPRACH_conf['sc_C1']],
                [NPRACH_conf['period_C2'], rep_C2, NPRACH_conf['sc_C2']]]
        
        _, _ = self.m.carrier_set.update_ra_parameters(CE_args_list, th_C1, th_C0)

        # configure RAR windows per CE and register NPRACH parameters info
        for ce in range(3):
            self.RAR_window_size[ce] = RAR_window_size

        # store NPRACH configuration
        self.NPRACH_conf = NPRACH_conf

        # re-schedule NPRACH_update
        next_time = self.event.t + self.NPRACH_update_period

        schedule_event(next_time, self.event)

        self.update_state()
    
    def msg3_outcome(self, ue, t, connected):
        '''
        method used by RxProcedure to notify the outcome of an msg3 transmission
        this represents the establishment of a ue connection
        '''
        w_id = ue.rar_w_id
        w_id_list = self.RAR_w_ids[ue.CE_level]
        i_ = 0
        window_active = w_id in w_id_list
        if window_active:
            i_ = w_id_list.index(w_id)
        if connected and ue.state != STATE.RAO:
            # t={self.time}: MSG3 received from UE {ue.id} --> UE connected
            ue.state = STATE.CONNECTED
            ue.t_connection = t # time stamp 
            self.m.population.msg4(ue.id) # notify population that this ue is connected
            self.connected_UEs[ue.id] = ue # store connected ue in ordered dict
            self.m.perf_monitor.register_ue(ue) # keep track of this ue for reward and performance
            # ---
            self.RAR_ids[ue.CE_level] += 1
            self.RAR_detected[ue.CE_level] += 1
            sample = -1 * ue.loss
            self.incoming.append(sample)
            self.distribution_update(sample)
            # ---
            # connected ues: {list(self.connected_UEs.keys())}
        elif window_active:            
            ue.state = STATE.RAR # there is another chance for this guy
            self.RAR_UEs[ue.CE_level][i_] += 1
        self.update_state()

    def NPUSCH_end(self, UE_id, t, received, sinr):
        '''
        method used by RxProcedure to notify the outcome of an NPUSCH
        '''      
        ue = self.scheduled_UEs.pop(UE_id)
        ue.sinr = sinr # for HARQ
        if received:
            bits = ue.bits
            ue.acked_data += bits
            ue.retx_buffer = 0
            ue.bits = 0
            ue.new_data = True
            self.m.perf_monitor.account_rx(t, ue)
            if not ue.buffer:
                ue.state = STATE.DONE
                ue.t_disconnection = t
                delay, service_time, _ = self.m.perf_monitor.unregister_ue(ue) # for reward and performance
                self.m.population.departure(ue) # returns to the source
                self.departures.append(delay) # for NPRACH statistics
                self.total_departures += 1 # total departures counter
                self.service_times.append(service_time)
            else:
                self.connected_UEs[ue.id] = ue # back to the queue
        else:
            ue.new_data = False
            ue.harq_rtx += 1
            ue.retx_buffer = ue.bits # retx_buffer tells what happened before
            self.m.perf_monitor.account_error(ue)
            self.connected_UEs[ue.id] = ue # back to the queue
        self.update_state()
    