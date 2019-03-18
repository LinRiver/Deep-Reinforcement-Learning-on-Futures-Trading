# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:36:17 2019

@author: fgw
"""

import pandas as pd

from collections import deque
from sklearn.preprocessing import MinMaxScaler

class Data_Process(object):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, sep=',')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

        TWSE_data = data[['Open', 'High', 'Low', 'Close', 'Futures spread', 'Jump spread']]

        Observe_data = TWSE_data.copy()
        Observe_data['High'] = Observe_data['High'].shift(1)
        Observe_data['Low'] = Observe_data['Low'].shift(1)
        Observe_data['Close'] = Observe_data['Close'].shift(1)

        price_data = data[['Open_futures', 'Close_futures']]
        price_data.columns = ['Open', 'Close']
        
        self.N_price_data = price_data
        self.N_Observe_data = Observe_data
        
    def spilt_to_train_test(self, date_split, date_end, data_start = '2006-01-01'):

        train_data = self.N_price_data[data_start:date_split]
        test_data = self.N_price_data[date_split:date_end]
        
        train_data_Ob = self.N_Observe_data[data_start:date_split]
        test_data_Ob = self.N_Observe_data[date_split:date_end]

        ##### train  

        train_data_Ob_OHLC = train_data_Ob[['Open', 'High', 'Low', 'Close']]
        train_data_Ob_spread = train_data_Ob[['Futures spread', 'Jump spread']]

        spread_scaler_train = MinMaxScaler(feature_range=(-1,1))

        Ntrain_data_Ob_spread = spread_scaler_train.fit_transform(train_data_Ob_spread)
        Ntrain_data_Ob_spread = pd.DataFrame(Ntrain_data_Ob_spread)
        Ntrain_data_Ob_spread.index = train_data_Ob_spread.index
        Ntrain_data_Ob_spread.columns = train_data_Ob_spread.columns

        Size = -23

        scalar_data = self.N_Observe_data[:data_start].iloc[Size:].values[:,:4]

        train_scalar_data = deque(maxlen=len(scalar_data)) 
        for i in range(len(scalar_data)):
            train_scalar_data.append(scalar_data[i])

        train_dic={}
        scaler_train = MinMaxScaler(feature_range=(-1,1))

        for i in range(len(train_data_Ob)):
            scaler_train.fit(train_scalar_data)
            train_dic[train_data_Ob_OHLC.index[i]] = scaler_train.transform(train_data_Ob_OHLC.iloc[i].values.reshape(1, -1)).reshape(-1)
            train_scalar_data.append(train_data_Ob_OHLC.iloc[i].values.reshape(-1))

        Ntrain_data_OHLC = pd.DataFrame(train_dic).transpose()
        Ntrain_data_OHLC.columns = train_data_Ob_OHLC.columns

        Ntrain_data = Ntrain_data_OHLC.join(Ntrain_data_Ob_spread)

        ##### test    

        test_data_Ob_OHLC = test_data_Ob[['Open', 'High', 'Low', 'Close']]
        test_data_Ob_spread = test_data_Ob[['Futures spread', 'Jump spread']]

        Ntest_data_Ob_spread = spread_scaler_train.transform(test_data_Ob_spread)
        Ntest_data_Ob_spread = pd.DataFrame(Ntest_data_Ob_spread)
        Ntest_data_Ob_spread.index = test_data_Ob_spread.index
        Ntest_data_Ob_spread.columns = test_data_Ob_spread.columns

        scalar_data = train_data_Ob.iloc[Size:].values [:,:4]

        test_scalar_data = deque(maxlen=len(scalar_data)) 
        for i in range(len(scalar_data)):
            test_scalar_data.append(scalar_data[i])

        test_dic={}
        scaler_test = MinMaxScaler(feature_range=(-1,1))

        for i in range(len(test_data_Ob)):
            scaler_test.fit(test_scalar_data)
            test_dic[test_data_Ob_OHLC.index[i]] = scaler_test.transform(test_data_Ob_OHLC.iloc[i].values.reshape(1, -1)).reshape(-1)
            test_scalar_data.append(test_data_Ob_OHLC.iloc[i].values.reshape(-1))

        Ntest_data_OHLC = pd.DataFrame(test_dic).transpose()
        Ntest_data_OHLC.columns = test_data_Ob_OHLC.columns

        Ntest_data = Ntest_data_OHLC.join(Ntest_data_Ob_spread)

        return train_data, Ntrain_data, test_data, Ntest_data

    def spilt_to_train_val_test(self, date_val_split, date_test_split, date_end, data_start = '2006-01-01'):
      
        train_data = self.N_price_data[data_start:date_val_split]
        val_data = self.N_price_data[date_val_split:date_test_split]
        test_data = self.N_price_data[date_test_split:date_end]

        train_data_Ob = self.N_Observe_data[data_start:date_val_split]
        val_data_Ob = self.N_Observe_data[date_val_split:date_test_split]
        test_data_Ob = self.N_Observe_data[date_test_split:date_end]

        ##### train  

        train_data_Ob_OHLC = train_data_Ob[['Open', 'High', 'Low', 'Close']]
        train_data_Ob_spread = train_data_Ob[['Futures spread', 'Jump spread']]

        spread_scaler_train = MinMaxScaler(feature_range=(-1,1))

        Ntrain_data_Ob_spread = spread_scaler_train.fit_transform(train_data_Ob_spread)
        Ntrain_data_Ob_spread = pd.DataFrame(Ntrain_data_Ob_spread)
        Ntrain_data_Ob_spread.index = train_data_Ob_spread.index
        Ntrain_data_Ob_spread.columns = train_data_Ob_spread.columns

        Size = -23

        scalar_data = self.N_Observe_data[:data_start].iloc[Size:].values[:,:4]

        train_scalar_data = deque(maxlen=len(scalar_data)) 
        for i in range(len(scalar_data)):
            train_scalar_data.append(scalar_data[i])

        train_dic={}
        scaler_train = MinMaxScaler(feature_range=(-1,1))

        for i in range(len(train_data_Ob)):
            scaler_train.fit(train_scalar_data)
            train_dic[train_data_Ob_OHLC.index[i]] = scaler_train.transform(train_data_Ob_OHLC.iloc[i].values.reshape(1, -1)).reshape(-1)
            train_scalar_data.append(train_data_Ob_OHLC.iloc[i].values.reshape(-1))

        Ntrain_data_OHLC = pd.DataFrame(train_dic).transpose()
        Ntrain_data_OHLC.columns = train_data_Ob_OHLC.columns

        Ntrain_data = Ntrain_data_OHLC.join(Ntrain_data_Ob_spread)

        ##### val

        val_data_Ob_OHLC = val_data_Ob[['Open', 'High', 'Low', 'Close']]
        val_data_Ob_spread = val_data_Ob[['Futures spread', 'Jump spread']]

        Nval_data_Ob_spread = spread_scaler_train.transform(val_data_Ob_spread)
        Nval_data_Ob_spread = pd.DataFrame(Nval_data_Ob_spread)
        Nval_data_Ob_spread.index = val_data_Ob_spread.index
        Nval_data_Ob_spread.columns = val_data_Ob_spread.columns

        scalar_data = train_data_Ob.iloc[Size:].values[:,:4]

        val_scalar_data = deque(maxlen=len(scalar_data)) 
        for i in range(len(scalar_data)):
            val_scalar_data.append(scalar_data[i])

        val_dic={}
        scaler_val = MinMaxScaler(feature_range=(-1,1))

        for i in range(len(val_data_Ob)):
            scaler_val.fit(val_scalar_data)
            val_dic[val_data_Ob_OHLC.index[i]] = scaler_val.transform(val_data_Ob_OHLC.iloc[i].values.reshape(1, -1)).reshape(-1)
            val_scalar_data.append(val_data_Ob_OHLC.iloc[i].values.reshape(-1))

        Nval_data_OHLC = pd.DataFrame(val_dic).transpose()
        Nval_data_OHLC.columns = val_data_Ob_OHLC.columns

        Nval_data = Nval_data_OHLC.join(Nval_data_Ob_spread)

        ##### test    

        test_data_Ob_OHLC = test_data_Ob[['Open', 'High', 'Low', 'Close']]
        test_data_Ob_spread = test_data_Ob[['Futures spread', 'Jump spread']]

        spread_scaler_test = MinMaxScaler(feature_range=(-1,1))
        spread_scaler_test.fit(train_data_Ob_spread.append(test_data_Ob_spread))

        Ntest_data_Ob_spread = spread_scaler_test.transform(test_data_Ob_spread)
        Ntest_data_Ob_spread = pd.DataFrame(Ntest_data_Ob_spread)
        Ntest_data_Ob_spread.index = test_data_Ob_spread.index
        Ntest_data_Ob_spread.columns = test_data_Ob_spread.columns

        scalar_data = val_data_Ob.iloc[Size:].values [:,:4]

        test_scalar_data = deque(maxlen=len(scalar_data)) 
        for i in range(len(scalar_data)):
            test_scalar_data.append(scalar_data[i])

        test_dic={}
        scaler_test = MinMaxScaler(feature_range=(-1,1))

        for i in range(len(test_data_Ob)):
            scaler_test.fit(test_scalar_data)
            test_dic[test_data_Ob_OHLC.index[i]] = scaler_test.transform(test_data_Ob_OHLC.iloc[i].values.reshape(1, -1)).reshape(-1)
            test_scalar_data.append(test_data_Ob_OHLC.iloc[i].values.reshape(-1))

        Ntest_data_OHLC = pd.DataFrame(test_dic).transpose()
        Ntest_data_OHLC.columns = test_data_Ob_OHLC.columns

        Ntest_data = Ntest_data_OHLC.join(Ntest_data_Ob_spread)

        return train_data, Ntrain_data, val_data, Nval_data, test_data, Ntest_data        