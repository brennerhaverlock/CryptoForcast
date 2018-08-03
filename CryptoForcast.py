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
       # init_data,init_label = None,None
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
batch_size = 100 #number of samples in batch
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
    
    
#tesnsor transformations-- function dynamic_runn requires output to be specific format

all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs], axis = 0 )

#make all outputs [seq_length, batch_size, num_nodes]

all_lstm_outputs, state = tf.nn.dynamic_rnn(
        drop_multi_cell, all_inputs, initial_state = tuple(initial_state),
        time_major = True, dtype = tf.float32)

#reshape outputs

all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrolling, num_node[-1]])
all_outputs = tf.nn.xw_plus_b(all_lstm_outputs,r_0,r_1)
split_outputs = tf.split(all_outputs, num_unrolling, axis = 0)

#calculate loss

print('Defining training loss')
loss = 0.0
with tf.control_dependencies([tf.assign(c_state[xi], state[xi][0]) for xi in range(n_layers)]+
                             [tf.assign(h_state[xi], state[xi][1]) for xi in range(n_layers)]):
    for ui in range(num_unrolling):
        loss += tf.reduce_mean(0.5*(split_outputs[ui] - train_outputs[ui]) ** 2)
        
print('Learning Rate decay operations')
global_step = tf.Variable(0, trainable = False)
inc_glstep = tf.assign(global_step, global_step + 1)
tf_learning_rate = tf.placeholder(shape = None, dtype = tf.float32)
tf_min_learning_rate = tf.placeholder(shape = None, dtype = tf.float32)

learning_rate = tf.maximum(
        tf.train.exponential_decay(tf_learning_rate,global_step, decay_steps=1,
                                   decay_rate= 0.5,
                                   staircase= True), tf_min_learning_rate)
print('TF Optimization Op')
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(
        zip(gradients, v))

print('\tAll Done')

#defining prediction tensorflow operations

print('Defining related TF Functions')

sample_inputs = tf.placeholder(tf.float32, shape = [1, D])

#Maintain LSTM state for prediction 

sample_c, sample_h, initial_sample_state = [],[],[]

for xi in range(n_layers):
    sample_c.append(tf.Variable(tf.zeros([1, num_node[xi]]), trainable = False))
    sample_h.append(tf.Variable(tf.zeros([1, num_node[xi]]), trainable = False))
    initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[xi], sample_h[xi]))
    
sample_state_reset = tf.group(*[tf.assign(sample_c[xi], tf.zeros([1, num_node[xi]])) for xi in range(n_layers)],
                              *[tf.assign(sample_h[xi], tf.zeros([1, num_node[xi]])) for xi in range(n_layers)])

sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs, 0),
                                                 initial_state = tuple(initial_sample_state),
                                                 time_major = True,
                                                 dtype = tf.float32)

with tf.control_dependencies([tf.assign(sample_c[xi],sample_state[xi][0]) for xi in range (n_layers)]+
                             [tf.assign(sample_h[xi], sample_state[xi][1]) for xi in range (n_layers)]):
    sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), r_0, r_1)
print('\tAll Done')

#Fun part of train and predict movements for several learning steps and see good/bad

epochs = 35 #learning  steps
valid_summary = 1 #interval for test predictions
 
n_predict_once = 50 #steps continously predicting for
train_seq_value = train_data.size # Full length/value of training data

train_mse_ot = [] #array of train losses
test_mse_ot = [] # array of test losses
predictions_over_time = [] # array of predictions

session = tf.InteractiveSession()

tf.global_variables_initializer().run()

#for decaying learning rate
loss_nondecrease_count = 0 
loss_nondecrease_threshold = 2 #if no increase in this many steps decrease learning rate

print('Init LSTM')
average_loss = 0 

#define data gen 
data_gen = DataGenSeq(train_data, batch_size, num_unrolling)

x_axis_seq = []

#points to start test predictions from
seq_test_points = np.arange(1200, 1500, 50).tolist()

for ep in range(epochs):
    # ==========================Training==========================
    
    for step in range(train_seq_value//batch_size):
        
        u_data, u_labels = data_gen.unroll_batches()
        
        feed_dict = {}
        for ui,(dat,lbl) in enumerate(zip(u_data, u_labels)):
            feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
            feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)
            
        feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate: 0.00001})
        _, 1 = session.run(([optimizer, loss]), feed_dict=feed_dict)
        
        average_loss += 1
        
#===============================VALIDATION========================================
        
    if (ep + 1) % valid_summary == 0:
        
        average_loss = average_loss/(valid_summary*(train_seq_value//batch_size))
        
        if (ep+1)%valid_summary == 0:
            print('Average loss at step %d: %f' % (ep + 1, average_loss))
            
        train_mse_ot.append(average_loss)
        
        average_loss = 0 #reset
        
        seq_predictions = []
        
        test_mse_loss_seq = []
        
        #====================== Updating State and Making Predictions ===========================
        
        for w_ix in test_points_seq:
            mse_test_loss = 0.0
            predictions = []
            
            if (ep + 1)-valid_summary==0:
                #calculate x_axis values in first valid epoch
                x_axis = []
                
            #Feed past values of stock prices to make predictions from there
            for re_i in range(w_ix-num_unrolling+1,w_ix-1):
                now_price = all_mid_data[re_i]
                dict_entry[sample_inputs] = np.array(now_price).reshape(1,1)
                _ = session.run(sample_prediction, dict_entry = dict_entry)
                
            dict_entry = {}
            
            now_price = all_mid_data[w_ix]
            
            dict_entry[sample_inputs] = np.array(now_price).reshape(1,1)
            
            #Making predictions for x steps each one uses previous as it's input
            
            for pred_ix in range(n_predict_once):
                
                pred = session.run(sample_prediction, dict_entry=dict_entry)
                
                predictions.append(np.asscalar(pred))
                
                dict_entry[sample_inputs] = np.asarray(pred).reshape(-1,1)
                
                if (ep+1) - valid_summary == 0:
                    #only calculate x_axis values in first epoch validation
                    x_axis.append(w_ix+pred_ix)
                
                mse_test_loss += 0.5*(pred-all_mid_data[w_ix+pred_ix])**2
            
            session.run(reset_sample_states)
            
            seq_predictions.append(np.array(predictions))
            
            mse_test_loss /= n_predict_once
            test_mse_loss_seq.append(mse_test_loss)
            
            if (ep + 1) - valid_summary == 0:
                x_axis_seq.append(x_axis)
                
        current_mse_test = np.mean(test_mse_loss_seq)
        
        #logic for learning rate decay 
        
        if len(test_mse_ot) > 0 and current_mse_test > min(test_mse_ot):
            loss_nondecrease_count += 1
        else:
            loss_nondecrease_count = 0
            
        if loss_nondecrease_count > loss_nondecrease_threshold :
            session.run(inc_glstep)
            loss_nondecrease_count = 0
            print('\t Decreasing learning rate by 0.5')
            
        test_mse_ot.append(current_mse_test)
        print('\tTest MSE %.5f'% np.mean(test_mse_loss_seq))
        predictions_over_time.append(seq_predictions)
        print('\t Finished Prediction')
















