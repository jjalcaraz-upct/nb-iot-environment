#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the elements needed to run an event-based simulator.
The simulator follows a publish-subscribe pattern.
Entities handling the events have to subscribe themselves to the handled events.

Created on Jan 14, 2022

@author: juanjosealcaraz

"""

from heapq import heappush, heappop
from itertools import count

NodeB_events = ['NPDCCH_arrival', 'NPRACH_update', 'RAR_window_end']
AccessProcedure_events = ['NPRACH_start', 'NPRACH_end']
RxProcedure_events = ['NPUSCH_end', 'Msg3']
Population_events = ['UE_arrival', 'Backoff_end', 'MAC_timer']

VOID = -1

subscribers = dict()

def subscribe(event_type: str, fn):
   if not event_type in subscribers:
      subscribers[event_type] = []
   subscribers[event_type] = fn

def post_event(event):
   if not event.type in subscribers:
      return
   fn = subscribers[event.type]
   fn(event)

class Event:
    def __init__(self, type: str, **kwargs):
        self.type = type
        self.__dict__.update(kwargs)
    def __repr__(self) -> str:
        # return 'Event '+' '.join('{}: {}'.format(k, v) for k, v in self.__dict__.items())
        return self.type
        
class FEL:
    '''Future Event List'''
    def __init__(self):
        self.events = []
        self.counter = count()  # unique sequence count

    def __repr__(self):
        output = 'fel = [ \n'
        for e in self.events:
            output +='      {} \n'.format(e)
        output +='      ]'
        return output
    
    def __setitem__(self, time, event):
        # events are planned with:
        # fel[time] = event
        count = next(self.counter)
        entry = [time, count, event]
        heappush(self.events, entry)
    
    def __len__(self):
        return len(self.events)

    def reset(self):
        self.events = []
        self.counter = count()  # unique sequence count        
    
    def pop_next(self):
        # returns next event
        if self.events:
            time, _, event = heappop(self.events)
            return time, event
        else:
            return VOID, Event('void')

    def time_next(self):
        # returns the time of the next event
        if self.events:
            time, _, _ = self.events[0]
            return time
        else:
            return VOID

fel = FEL()

def schedule_event(time, event):
    event.t = time # add timestamp to every event
    fel[time] = event

# def schedule_event_old(time, event):
#     fel[time] = event

def run_until(events):
    # advances the FEL until reaching any event in events
    time, event = fel.pop_next()
    while event.type not in events and len(fel)>0:
        post_event(event)
        time, event = fel.pop_next()
        print(f't: {time}, event type: {event.type}, len(fel): {len(fel)}')
    return time, event

def run_until_t(t):
    # advances the FEL until reaching a given time t
    time = fel.time_next()
    while time <= t and len(fel)>0:
        time , event = fel.pop_next()
        post_event(event)       
    return time

def run_until_t_debug(t):
    # advances the FEL until reaching a given time t
    time = fel.time_next()
    while time <= t and len(fel)>0:
        time , event = fel.pop_next()
        print(f't: {time}, event type: {event.type}')
        post_event(event)       
    return time

def run():
    while len(fel)>0:
        _, event = fel.pop_next()
        post_event(event)

def run_debug():
    while len(fel)>0:
        t, event = fel.pop_next()
        print(f't: {t}, event type: {event.type}')
        post_event(event)

def print_fel():
    print(fel)

if __name__ == '__main__':
    class NodeB:
        def __init__(self):
            subscribe('arrival', self.fn)
            subscribe('departure', self.fnn)

        def fn(self, event):
            print(event)

        def fnn(self, event):
            print(event)

    nb = NodeB()
    
    subscribe('fistro', print)

    e1 = Event('arrival', a = 2)
    e2 = Event('departure', a = 5, b = 44)
    e3 = Event('fistro', j = 'pecadorr')

    schedule_event(3, e1)
    schedule_event(7, e2)
    schedule_event(4, e3)

    print(fel)

    time , event = run_until(['departure'])

    print(event)
