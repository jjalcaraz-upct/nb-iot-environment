#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Implements the Node B, which the main entity of the system.
The node b is in charge, among other things, of advancind the discrete-event simulation process.

Created on Jan 13, 2022

@author: juanjosealcaraz

"""

from event_manager import schedule_event, post_event, fel, Event
from action_reader import ActionReader
from operator import attrgetter
import numpy as np
import pickle
from parameters import *
from user import STATE
from utils import find_next_integer_index

# DEBUG USE ONLY
tx_ues = []

# Node B states and processed events
NodeB_states = ['Scheduling', 'NPRACH_update', 'RAR_window', 'RAR_window_end','Idle']
NodeB_events = ['NPDCCH_arrival', 'NPRACH_update', 'RAR_window_end']

# action reader
reader = ActionReader(NPDCCH_period)

# auxiliary function
def average(lst):
    if len(lst) > 0:
        return round(sum(lst) / len(lst), 2)
    else:
        return 0
    
# auxiliary function
def find_next_integer_index(lst, n):
    """
    Finds the smallest integer in the list 'lst' that is greater or equal than integer 'n' and its position
    """
    greater_integers = [(i, j) for j, i in enumerate(lst) if i >= n]
    if greater_integers:
        return min(greater_integers)
    else:
        return max(lst), len(lst)-1
    
def itbs_bl(itbs, bl):
    ''' returns the block size closest to bl using itbs or the most similar one'''
    imcs = Itbs_to_Imcs[itbs]
    for i in range(imcs, 7):
        itbs = Imcs_to_Itbs[i]
        lst = tbs_lists[itbs]
        block_size, pos = find_next_integer_index(lst,bl)
        if block_size:
            return itbs, block_size, N_ru_list[pos]

class NodeB:
    '''
    Applies the actions at each step which corresponds either to a NPDCCH arrival or a NPRACH_update.
    It is instantiated with a message switch that connects it to the other entities.
    Handled events: NPDCCH_arrival, NPRACH_update
    Scheduled events: NPUSCH_end, RAR_window_end, NPRACH_update
    '''
    def __init__(self, m, sc_adjustment = False, mcs_automatic = False, sort_criterium = 't_connection', tx_all_buffer = True):
        self.m = m # message switch
        self.m.set_node(self)

        # n carriers determined by the carrier set object
        self.n_carriers = self.m.carrier_set.n_carriers

        # NPDCCH parameters
        self.NPDCCH_sf_left = NPDCCH_sf

        # NPRACH parameters
        self.NPRACH_update_period = NPRACH_update_period
        self.NPRACH_conf = {
            0: {
                'periodicity': 0,
                'subcarriers': 0,
                'repetitions': 0
            },
            1: {
                'periodicity': 0,
                'subcarriers': 0,
                'repetitions': 0
            },
            2: {
                'periodicity': 0,
                'subcarriers': 0,
                'repetitions': 0
            },
            'p_anchor': 0,
            'rar_window': 0
        }

        # initialize storage structures
        self.reset_storage()

        # RAR parameters
        self.RAR_window_size = [2 * NPDCCH_period]*3 # can be between 2 and 10 (NPDCCH periods)

        # Node B parameters
        self.state = 'NPRACH_update'
        self.time = 0
        self.event = Event('NPRACH_update', t = 0)

        # optional configuration
        self.sc_adjustment = sc_adjustment
        self.mcs_automatic = mcs_automatic
        self.sort_criterium = sort_criterium
        self.tx_all_buffer = tx_all_buffer
        if mcs_automatic:
            self.loss_to_mcs = pickle.load(open("loss_to_mcs.p", "rb"))

        # step methods for each state
        self.step_methods = {
            'Scheduling': self.step_data,
            'RAR_window': self.step_RAR_window, 
            'RAR_window_end': self.step_RAR_end,
            'NPRACH_update': self.step_update,
        }

    def reset_storage(self):
        # NPRACH detections
        self.NPRACH_detections = [[], [], []]

        # RAR parameters
        self.RAR_UEs = [0, 0, 0] # list with the number of UE preambles received at each CE level
        self.RAR_active_window = [False, False, False]

        # Connected UEs
        self.connected_UEs = dict()
        self.selectable_UE_ids = []

        # Scheduled UEs
        self.scheduled_UEs = dict()

        # Served UEs (for debug purposes)
        self.served_UEs = []


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
        obs, info = self.get_node_state_and_info()
        return np.array(obs, dtype=float), info


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
                if self.check_event(event.type):
                    break
            else:
                post_event(event)


    def start_RAR_window(self, number_of_UEs, CE_level, time):
        ''' 
        Method invoked by AccessProcedure to notify the start of the RAR window,
        number_of_UEs: number of detected preambles
        CE_level: CE level of the RAR window
        '''
        # update RAR state
        self.RAR_UEs[CE_level] = number_of_UEs
        self.NPRACH_detections[CE_level].append(number_of_UEs)
        self.RAR_active_window[CE_level] = True
        if number_of_UEs > 0:
            self.state = 'RAR_window'
        
        # schedule end of RAR window for this CE_level
        e_time = time + self.RAR_window_size[CE_level]
        RAR_window_end = Event('RAR_window_end', t = e_time, CE_level = CE_level)
        schedule_event(e_time, RAR_window_end)


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
        obs, info = self.get_node_state_and_info()

        # get reward
        reward = self.m.perf_monitor.get_reward()
        perf_info = self.m.perf_monitor.get_info()
        info.update(perf_info)
        return np.array(obs, dtype=float), reward, False, info


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
        if sum(self.RAR_active_window) and sum(self.RAR_UEs)>0:
            self.state = 'RAR_window'
        elif len(self.connected_UEs) > 0:
            self.state = 'Scheduling'
        else:
            self.state = 'Idle'         

    def get_node_state_and_info(self):
        '''
        extracts the full state of the node:
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
        info = {'time': self.time, 'state': self.state, 'total_ues': total_ues}
        if self.state in ['Scheduling','RAR_window_end']: # no RAR window
            selectable_ues = []
            sf_left = self.NPDCCH_sf_left
            max_CE = CE_for_NPDCCH_sf_left[sf_left]
            n_ue = 0       
            for ue in sorted(self.connected_UEs.values(), key=attrgetter(self.sort_criterium)):
                if ue.CE_level <= max_CE:
                    selectable_ues.append(ue)
                    n_ue += 1
                if n_ue == N_users: # N_users in parameters.py: the n users observed  
                    break
            state = ue_state(self.time, selectable_ues, total_ues)
            self.selectable_UE_ids = [ue.id for ue in selectable_ues] # selectable ues until next decision
            info['ues'] = self.selectable_UE_ids

        elif self.state == 'RAR_window': # RAR window
            sf_left = self.NPDCCH_sf_left
            max_CE = CE_for_NPDCCH_sf_left[sf_left]
            ues_per_CE = [0, 0, 0]
            for ce, (ues, active) in enumerate(zip(self.RAR_UEs, self.RAR_active_window)):
                if ce <= max_CE and active:
                    ues_per_CE[ce] = ues
            state = rar_state(ues_per_CE)
            
        else: # NPRACH update
            state = nprach_state(self.NPRACH_conf, self.NPRACH_detections, self.n_carriers)
            self.NPRACH_detections = [[], [], []] # reset
            info['NPRACH'] = self.NPRACH_conf 
        
        for c in range(self.n_carriers):
            state.extend(self.m.carrier_set.get_carrier_state(c)) 

        return state, info


    def schedule_NPDCCH_arrival(self, elapsed_sfs):
        '''
        Auxiliary method for re-scheduling a NPDCCH_arrival event after processing one
        '''
        self.NPDCCH_sf_left = max(self.NPDCCH_sf_left - elapsed_sfs, 0)

        # re-schedule NPDCCH_arrival
        next_time = self.time + elapsed_sfs
        if not self.NPDCCH_sf_left:
            next_time += NPDCCH_period
            self.NPDCCH_sf_left = NPDCCH_sf

        schedule_event(next_time, self.event)


    def allocate_resources(self, c_id, ref_t, delay, N_sc, N_ru, N_rep, CE_level = -1, UE_id = -1, I_tbs = -1):
        '''
        allocate resources in the selected carrier with a given delay and returns the end-of-tx time
        if the resources do not fit with the selected number of subcarriers it retries with a different number
        '''
        n_sc = N_sc
        N_sf = N_sf_per_ru[N_sc] * N_ru * N_rep
        t_ul_end = self.m.carrier_set.allocate_resources(c_id, ref_t, delay = delay, N_sc = N_sc, N_sf = N_sf, CE_level = CE_level, UE_id = UE_id, I_tbs = I_tbs, N_rep = N_rep)
        if not t_ul_end and self.sc_adjustment: # automatic adjustment of subcarriers
            for n_sc in N_sc_selection_list[N_sc]:
                N_sf = N_sf_per_ru[n_sc] * N_ru * N_rep
                t_ul_end = self.m.carrier_set.allocate_resources(c_id, ref_t, delay = delay, N_sc = n_sc, N_sf = N_sf, CE_level = CE_level, UE_id = UE_id, I_tbs = I_tbs, N_rep = N_rep)
                if t_ul_end:
                    return t_ul_end, n_sc, N_sf
        return t_ul_end, n_sc, N_sf


    def step_RAR_window(self, action):
        '''
        Schedules an UL grant for the msg3 of one user
        '''
        ce_lists = [[0,2,1],[1,2,0],[2,1,0]]

        # extract the parameters from the action
        RAR_params = reader.scheduling_action(action, RAR_action = True)

        # extract the CE_level
        CE_level = RAR_params['i']
        ce = min(CE_level, 2)

        # check if this control is useful and, if not, try with the others
        valid_control = False
        for CE_level in ce_lists[ce]:
            NPDCCH_sf = NPDCCH_sf_per_CE[CE_level] # required NPDCCH repetitions
            valid_control = self.RAR_UEs[CE_level] > 0 and self.NPDCCH_sf_left >= NPDCCH_sf
            if valid_control:
                break

        # no valid control found
        if not valid_control:
            self.schedule_NPDCCH_arrival(self.NPDCCH_sf_left)          
            return

        # extract every required parameter
        carrier_id = RAR_params['carrier']
        I_tbs = RAR_params['I_tbs']
        delay = RAR_params['delay']
        N_sc = RAR_params['N_sc']
        N_rep = RAR_params['N_rep']
        N_ru = RAR_params['N_ru']

        # check if the allocated resources fit into the selected carrier
        reference_time = self.time + NPDCCH_sf
        t_ul_end, N_sc, _ = self.allocate_resources(carrier_id, reference_time, delay, N_sc, N_ru, N_rep, CE_level = CE_level)
        # if the resources fit, extract the UE parameters from the action and submit an UL grant for msg3
        if t_ul_end:     
            self.m.population.msg3_grant(CE_level, reference_time, t_ul_end, I_tbs = I_tbs, N_rep = N_rep)
            self.RAR_UEs[CE_level] -= 1 

        # re-schedule NPDCCH
        self.schedule_NPDCCH_arrival(NPDCCH_sf)
        self.update_state()


    def step_RAR_end(self, action):
        '''
        Processes a 'RAR_window_end' event associated to a given CE_level.
        Communicates the backoff value to the UEs
        '''

        backoff = reader.UE_contention_resolution(action)

        CE_level = self.event.CE_level
        t = self.event.t

        self.RAR_UEs[CE_level] = 0
        self.RAR_active_window[CE_level] = False

        self.m.population.RAR_window_end(t, CE_level, backoff)
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
            # nothing to do here: re-schedule NPDCCH
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
            data_params = reader.scheduling_action(action, I_tbs = I_tbs, N_rep = N_rep)
        elif self.mcs_automatic: # for automatic configuration we use predefined I_tbs and N_rep
            loss = round(ue.loss, 1)
            loss = max(121.4, min(171.4, loss))
            (I_tbs, N_rep) = self.loss_to_mcs[loss]
            data_params = reader.scheduling_action(action, I_tbs = I_tbs, N_rep = N_rep)

        # get the required number of NPDCCH repetitions
        NPDCCH_sf = NPDCCH_sf_per_CE[CE_level]

        # extract every required parameter
        carrier_id = data_params['carrier']
        delay = data_params['delay']
        N_sc = data_params['N_sc']
        N_rep = data_params['N_rep']
        I_tbs = data_params['I_tbs']
        I_N_ru = data_params['N_ru'] # this is an index in the tbs_list

        if self.tx_all_buffer: 
            # the N_ru is determined automatically from the I_tbs and the buffer size
            I_tbs, block_size, N_ru = itbs_bl(I_tbs, ue.buffer) # upgrades the I_tbs if required to fit the buffer and gets the block size
        else:
            lst = tbs_lists[I_tbs]
            block_size, pos = find_next_integer_index(lst, ue.buffer)
            if pos < I_N_ru:
                N_ru = N_ru_list[pos]
            else:
                N_ru = N_ru_list[I_N_ru]
                block_size = lst[I_N_ru]

        # check if the allocated resources fit into the selected carrier
        reference_time = self.time + NPDCCH_sf
        t_ul_end, N_sc, _ = self.allocate_resources(carrier_id, reference_time, delay, N_sc, N_ru, N_rep, UE_id = ue_id, I_tbs = I_tbs)
        # if the resources fit, submit an UL grant for PUSCH
        if t_ul_end:
            self.data_grant(ue_id, t_ul_end, I_tbs = I_tbs, N_rep = N_rep, tbs = block_size)
        elif new_data:
            self.m.perf_monitor.account_unfit(ue)
        
        # re-schedule NPDCCH
        self.schedule_NPDCCH_arrival(NPDCCH_sf)
        self.update_state()


    def data_grant(self, UE_id, t_end, I_tbs = 1, N_rep = 2, tbs = 256):
        '''
        method to assign an UL transmission grant to a specific UE
        '''
        ue = self.connected_UEs.pop(UE_id) # extract the ue 
        
        ue.I_tbs = I_tbs
        ue.N_rep = N_rep
        ue.tbs = tbs # used by the channel to compute the error probability

        if ue.retx_buffer > 0: # only one HARQ process
            ue.bits = ue.retx_buffer

        elif tbs >= ue.buffer: # transmit all the buffered bits
            ue.bits = ue.buffer
            ue.buffer = 0

        else:
            ue.bits = min(ue.buffer, tbs)
            ue.buffer -= ue.bits
        
        self.scheduled_UEs[ue.id] = ue
        
        e = Event('NPUSCH_end', ue = ue) # schedule NPUSCH_end event (handled by RxProcedure)
        schedule_event(t_end, e)


    def step_update(self, action):
        '''
        Updates the NPRACH parameters
        '''

        # extract parameters
        RAR_window_size, UE_kwargs, CE_args_list, rar_window_i = reader.NPRACH_update_action(action)

        # inform the UEs
        self.m.population.update_ra_parameters(**UE_kwargs)

        # configure carrier
        periods, _, N_sfs = self.m.carrier_set.update_ra_parameters(CE_args_list)

        # register NPRACH conf parameters for observation
        for ce in range(3):
            max_sfs = periods[ce] - 2*N_sfs[ce]
            max_periods = np.floor(max_sfs/NPDCCH_period)
            self.RAR_window_size[ce] = min(int(max_periods*NPDCCH_period), RAR_window_size)
            self.NPRACH_conf[ce]['periodicity'] = CE_args_list[ce][0]
            self.NPRACH_conf[ce]['repetitions'] = CE_args_list[ce][1]
            self.NPRACH_conf[ce]['subcarriers'] = CE_args_list[ce][2]
        self.NPRACH_conf['p_anchor'] = UE_kwargs['probability_anchor']
        self.NPRACH_conf['rar_window'] = rar_window_i

        # re-schedule NPRACH_update
        next_time = self.event.t + self.NPRACH_update_period

        schedule_event(next_time, self.event)
        self.update_state()
    

    def msg3_outcome(self, ue, t):
        '''
        method used by RxProcedure to notify the outcome of an msg3 transmission
        this represents the establishment of a ue connection
        '''
        ue.state = STATE.CONNECTED
        ue.t_connection = t # time stamp 
        self.m.population.msg4(ue.id) # notify population that this ue is connected
        self.connected_UEs[ue.id] = ue # store connected ue in ordered dict
        self.m.perf_monitor.register_ue(ue) # keep track of this ue for reward and performance
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
                self.served_UEs.append(ue) # for debuging purposes
                self.m.perf_monitor.unregister_ue(ue) # for reward and performance
                self.m.population.departure(ue) # returns to the source
            else:
                self.connected_UEs[ue.id] = ue # back to the queue
        else:
            ue.new_data = False
            ue.harq_rtx += 1
            ue.retx_buffer = ue.bits # retx_buffer tells what happened before
            self.m.perf_monitor.account_error(ue)
            self.connected_UEs[ue.id] = ue # back to the queue
        self.update_state()
    
    def detailed_report(self):
        print('POPULATION UEs:')
        self.m.population.report_ue_states()

        print('Served ues')
        for ue in self.served_UEs:
            print(f'ue id: {ue.id}')
            print(f' > ue loss: {round(ue.loss,1)}, I_tbs: {ue.I_tbs}, N_rep: {ue.N_rep}')
            print(f' > t_arrival: {ue.t_arrival}')
            print(f' > t_contention: {ue.t_contention}')
            print(f' > t_connection: {ue.t_connection}')
            print(f' > t_disconnection: {ue.t_disconnection}')
            print(f' > backoff_counter: {ue.backoff_counter}')
            print(f' > timeout_counter: {ue.timeout_counter}')
            print(f' > retransmissions: {ue.harq_rtx}')
            print('')

        print('Scheduled UEs:')
        for ue in self.scheduled_UEs.values():
            print(f'ue id: {ue.id}')
            print(f' > t_arrival: {ue.t_arrival}')
            print(f' > t_contention: {ue.t_contention}')
            print(f' > t_connection: {ue.t_connection}')
            print('')

        print('Connected UEs:')
        for ue in self.connected_UEs.values():
            print(f'ue id: {ue.id}')
            print(f' > t_arrival: {ue.t_arrival}')
            print(f' > t_contention: {ue.t_contention}')
            print(f' > t_connection: {ue.t_connection}')
            print('')
    
    def brief_report(self):
        print(f'  Connected UEs: {len(self.connected_UEs)}')
        print('')
        print(f'  Scheduled UEs: {len(self.scheduled_UEs)}')
        print('')
        print(f'  Served UEs: {len(self.served_UEs)}')
        print('')
        print('PERFORMANCE METRICS: ')
        print('')
        arrivals, departures, backoffs = self.m.perf_monitor.get_throughput()
        print(f' >> arrivals: {sum(arrivals)}, departures: {sum(departures)}')
        print('')
        print(f' >> backoffs: {sum(backoffs)}')
        print('')
        histories = self.m.perf_monitor.get_histories()
        print(f' >> access time: {average(histories[0])}, tx time: {average(histories[1])}, total delay: {average(histories[2])}')
        print('')
        errors, attempts = self.m.perf_monitor.get_error_count()
        print(f' >> errors: {errors}, attempts: {attempts}')
        print('')