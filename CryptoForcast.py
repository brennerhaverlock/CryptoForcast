# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:25:45 2018

@author: Brenner
"""

#from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import config
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

data_source = 'alphavantage'
# For future data IF statement for API 
if data_source == 'alphavantage':
    
    api_key = config.api_key
    
    symbol = 'BTC'
    market = 'USD'
    
    url_string = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=%s&market=USD&apikey=%s"%(symbol,api_key)
    
    file_to_save = 'currency_daily_BTC-%s.csv'%symbol
    #If file for csv does not exist it will turn it into pandas dataframe and save
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            #extract data
            data = data['Time Series (Digital Currency Daily)']
            df = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap(USD)'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['1a. open (USD)']), float(v['2a. high (USD)']),
                            float(v['3a. low (USD)']), float(v['4a. close (USD)']), float(v['5. volume']), float(v['6. market cap (USD)'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        print('Data saved to : %s'%file_to_save)
        df.to_csv(file_to_save)
        
        #if data is already there load it from CSV
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)
#df = panda dataframe for local use
df = pd.read_csv('currency_daily_BTC-BTC.csv')        
df = df.sort_values('Date')
#Plot
plt.figure(figsize = (18, 9))
plt.plot(range(df.shape[0]), (df['Low']+ df['High'] /2.0))
#plt.plot(range(df.shape[]))
plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation = 45)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Mid Price', fontsize = 18)
plt.show()
#Setting high/low and mid prices for data
high_prices = df.loc[:,'High'].as_matrix()
low_prices = df.loc[:,'Low'].as_matrix()
mid_prices = (high_prices+low_prices)/2.0
#split 
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(mid_prices, test_size=0.2)

##Now we Scale the data to be between 0 and 1 
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

#Train the sclaer with training data and smooth data
smooth_window_size = 300
for di in range(0, 1200, smooth_window_size):
    scaler.fit(train_data[di:di+smooth_window_size,:])
    train_data[di:di+smooth_window_size,:] = scaler.transform(train_data[di:di+smooth_window_size,:])
    
#Normalize the last of the remaining data
scaler.fit(train_data[di+smooth_window_size:,:])
train_data[di+smooth_window_size:,:] = scaler.transform(train_data[di+smooth_window_size:,:])

#Reshape train and test data to the shape of data size
train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

#exponetial moving average smoothing for smooth curve vs ragged data
EMA = 0
gamma = 0.1

for t in range (1200):
    EMA = gamma*train_data[t] + (1 - gamma)*EMA
    train_data[t] = EMA 
    
# visual and test purposes
all_mid_data = np.concatenate([train_data,test_data], axis = 0)

#Using MSE (Mean Squared Error)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_ix in range(window_size,N):
    
    if pred_ix >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days = 1)
    else:
        date = df.loc[pred_ix,'Date']
        
    std_avg_predictions.append(np.mean(train_data[pred_ix-window_size:pred_ix]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_ix]) **2)
    std_avg_x.append(date)
print('MSE error for standard AVG: %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), all_mid_data, color = 'b', label = 'True')
plt.plot(range(window_size,N), std_avg_predictions, color = 'r', label = "Prediction")
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize = 18)
plt.show()

#Exponential Moving Average 

window_size = 100 

N = train_data.size

run_avg_predictions = []
run_avg_x = []
mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_ixx in range(1,N):
    
    running_mean = running_mean*decay + (1.0 - decay) * train_data[pred_ixx - 1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1] - train_data[pred_ixx])**2)
    run_avg_x.append(date)
    
print('MSE error for EMA average %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), all_mid_data, color = 'b', label = "True")
plt.plot(range(0,N), run_avg_predictions, color = 'orange', label = 'Prediction')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize = 18)
plt.show()

#Data gen to train model that will output input data 
class DataGenSeq(object):
    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size 
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]
    
    #takes in first batch of data
    def next_batch(self):
        
        batch_data = np.zeros((self._batch_size), dtype = np.float32)
        batch_labels = np.zeros((self._batch_size),dtype = np.float32)
        
        for b in range(self._batch_size):
            if self._cursor[b]+1>= self._prices_length:
                self._cursor[b] = np.random.randint(0,(b+1)* self._segments)
            
            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b]+np.random.randint(0,5)]
            
            self._cursor[b] = (self._cursor[b] + 1)% self._prices_length
        return batch_data, batch_labels
    
    #unroll data and move onto reset
    
    def unroll_batches(self):
        
        unroll_data,unroll_labels = [],[]
        init_data,init_label = None,None
        for ui in range(self._num_unroll):
            
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)
            
        return unroll_data,unroll_labels
    
    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length - 1))

dg = DataGenSeq(train_data,5,5)          
u_data, u_labels = dg.unroll_batches()

for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ', dat )
    
D = 1 #Dimensionality of the data this is 1d data
num_unrolling = 50 #Number of steps into future 
batch_size = 500 #number of samples in batch
num_node = [200,200,150] #Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_node)
dropout = 0.2 

tf.reset_default_graph()

#Input data

train_inputs, train_outputs = [],[]

#Unroll input over time defining placeholders for time step
for ui in range(num_unrolling):
    train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, D], name = 'train_inputs_%d'%ui))
    train_outputs.append(tf.placeholder(tf.float32, shape = [batch_size, 1], name = 'train_outputs%d'%ui))
#init the lstm cells 
lstm_cells = [
        
        tf.contrib.rnn.LSTMCell(num_units=num_node[ix],
                                state_is_tuple= True,
                                initializer = tf.contrib.layers.xavier_initializer()
                                )
        for ix in range(n_layers)]

drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
        lstm, input_keep_prob=1.0, output_keep_prob= 1.0 - dropout, state_keep_prob= 1.0 - dropout) for lstm in lstm_cells]
drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
#Linear regression layer w and b 
r_0 = tf.get_variable('w', shape=[num_node[-1], 1], initializer= tf.contrib.layers.xavier_initializer())
r_1 = tf.get_variable('b', initializer= tf.random_uniform([1], -0.1, 0.1))

#creating cell state and hidden state variables to keep the state of LSTM in check

c_state, h_state = [], []
initial_state = []
for ix in range(n_layers):
    c_state.append(tf.Variable(tf.zeros([batch_size, num_node[ix]]), trainable= False))
    h_state.append(tf.Variable(tf.zeros([batch_size, num_node[ix]]), trainable= False))
    initial_state.append(tf.contrib.rnn.LSTMStateTuple(c_state[ix], h_state[ix]))
    
    
            
        















