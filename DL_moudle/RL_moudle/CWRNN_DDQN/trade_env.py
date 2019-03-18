# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:07:26 2019

@author: fgw
"""

import numpy as np

class environment(object):
    
    def __init__(self, trade_data, feature_data, split_price=300, commodity_spot=200, initial_cash=1000000, reward_anchored=False, reward_scaler=100):
        self.data_close = trade_data['Close']
        self.data_open = trade_data['Open']
        self.data_observation = np.asarray(feature_data)
        self.action_space = ['long', 'short', 'close']
        self.split_price = split_price
        self.commodity_spot = commodity_spot
        self.initial_cash = initial_cash 
        self.reward_anchored = reward_anchored
        self.reset() 
        
        if self.reward_anchored == False:
            self.reward_scaler = reward_scaler
        else:
            pass

    def reset(self):
        self.step_counter = 0
        self.cash = self.initial_cash
        self.total_value = self.cash
        self.flags = 0
        self.hold_period = 0
        self.last_flag = 0
        self.stop_loss = False
        if self.reward_anchored == True: 
            self.reward_record = np.array([])
            
    def get_initial_state(self):
        observation=np.hstack((self.data_observation[0,:], self.flags)).reshape(-1, self.data_observation.shape[1]+1)
        return observation
        
    def get_action_space(self):
        return self.action_space

    def long(self):
        self.flags = 1 
        self.cash -= self.split_price 
        self.cost=self.data_open[self.step_counter]
        
    def short(self):
        self.flags = -1
        self.cash -= self.split_price 
        self.cost=self.data_open[self.step_counter]
        
    def keep(self):
        pass
        
    def close_long(self): 
        self.flags = 0
        self.profit=(self.data_open[self.step_counter]-self.cost) * self.commodity_spot 
        self.cash += (-self.split_price+self.profit) 

    def close_short(self):
        self.flags = 0
        self.profit=(self.cost-self.data_open[self.step_counter]) * self.commodity_spot
        self.cash +=  (-self.split_price+self.profit)

    def step_op(self, action):
        if action == 'long':
            if self.flags == 0:
                self.long()
            elif self.flags == -1: 
                self.close_short() 
                self.long()
            else:
                self.keep() 
        
        elif action == 'close':
            if self.flags == 1:
                self.close_long()
            elif self.flags == -1:
                self.close_short()
            else:
                pass
                
        elif action == 'short':
            if self.flags == 0:
                self.short()
            elif self.flags == 1:
                self.close_long()
                self.short()
            else:
                self.keep()
                
        else:
            raise ValueError("action should be elements of ['long', 'short', 'close']")
            
        if self.flags==1: 
            openposition=(self.data_close[self.step_counter]-self.cost) * self.commodity_spot 
        elif self.flags==-1:
            openposition=(self.cost-self.data_close[self.step_counter]) * self.commodity_spot
        else:
            openposition=0
            
        reward = self.cash + openposition - self.total_value 
        self.step_counter += 1 
        self.total_value = openposition + self.cash 
        
        if self.flags == self.last_flag and self.last_flag != 0:
            self.hold_period += 1
        else:
            self.hold_period = 0
        
        self.stop_loss= False
        
        if self.hold_period == 3: 
            if openposition < 0:
                self.stop_loss = True
            else :
                pass
        else:
            pass
        
        done = False
        
        if self.step_counter >= (len(self.data_close)-1): 
            done = True
    
        try:
            next_observation = np.hstack((self.data_observation[self.step_counter,:], self.flags)).reshape(-1, self.data_observation.shape[1]+1) 
        except:
            next_observation = None
            done = True
            #print('last trade for test data')
            
        self.last_flag=self.flags
        
        if self.reward_anchored == True:
            self.reward_record = np.hstack((self.reward_record, reward))
            return (reward-np.mean(self.reward_record))/(max(self.reward_record)-min(self.reward_record)+1e-7), next_observation, done 
        else:
            return reward/(self.commodity_spot*self.reward_scaler), next_observation, done 
    
    def step(self, action):
        if action == 0:
            return self.step_op('long')
        elif action == 1:
            return self.step_op('short')
        elif action == 2:
            return self.step_op('close')
        else:
            raise ValueError("action should be one of [0,1,2]")