# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:07:09 2019

@author: fgw
"""

import os
import tensorflow as tf
import numpy as np

from collections import deque

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

np.random.seed(1)
tf.set_random_seed(2)

class ClockworkRNN(object):
    def __init__(self, input_state, in_length, in_width, hidden_neurons, Rb, clockwork_periods, trainable=True):
        
        self.in_length = in_length #time_steps
        self.in_width = in_width #feature_dim  
        
        self.inputs = input_state
        self.hidden_neurons = hidden_neurons
        if hidden_neurons / Rb >= 2:
            self.Rb = Rb   
        else:
            raise ValueError("Rb must be less than half of hidden_neurons")
        self.clockwork_periods = clockwork_periods #for sequential data, this is able to memory at different timing, 1, 2, 4, 8, 16, 32 for instance 
        self.Ti_sum = len(self.clockwork_periods)
        
        self.trainable = trainable
        
    #mask_lower_traingular matrix is for forming mask_upper_traingular matrix by g-models
    def _Mask_Matrix(self,W,k):
        length = np.int(W/k)
        tmp = np.ones([W,W])
        for i in range(length)[1:]:
            tmp[i*k:(i+1)*k,:i*k]=0
        tmp[(i+1)*k:,:i*k]=0
        return np.transpose(tmp)
    
    def inference(self):
        #RNN initial weights
        with tf.variable_scope('input_layers1'):
            self.WI1 = tf.get_variable('WI1', shape=[self.in_width, self.hidden_neurons], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=self.trainable)
            self.bI1 = tf.get_variable('bI1', shape=[self.hidden_neurons], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=self.trainable)
            
        with tf.variable_scope('input_layers2'):
            self.WI2 = tf.get_variable('WI2', shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=self.trainable)
            self.bI2 = tf.get_variable('bI2', shape=[self.hidden_neurons], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=self.trainable)
        
        traingular_mask = self._Mask_Matrix(self.hidden_neurons, self.Rb)
        self.traingular_mask = tf.constant(traingular_mask, dtype=tf.float32, name='mask_upper_traingular')
        
        with tf.variable_scope('hidden_layers_1'):
            self.WH1 = tf.get_variable('WH1', shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=self.trainable)
            self.WH1 = tf.multiply(self.WH1, self.traingular_mask)
            self.bH1 = tf.get_variable('bH1', shape=[self.hidden_neurons], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=self.trainable)
            
        with tf.variable_scope('hidden_layers_2'):
            self.WH2 = tf.get_variable('WH2', shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=self.trainable)
            self.WH2 = tf.multiply(self.WH2, self.traingular_mask)
            self.bH2 = tf.get_variable('bH2', shape=[self.hidden_neurons], initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=self.trainable)

        #make training data structure transform to list structure 
        X_list = [tf.squeeze(x, axis=[1]) for x 
                  in tf.split(value=self.inputs, axis=1, num_or_size_splits=self.in_length, name='inputs_list')]
        
        with tf.variable_scope('clockwork_rnn') as scope:
            # set initial numbers on hidden layer 
            self.state1 = tf.get_variable('hidden_sate1', shape=[1, self.hidden_neurons],initializer=tf.zeros_initializer(), trainable=False)
            self.state2 = tf.get_variable('hidden_sate2', shape=[1, self.hidden_neurons],initializer=tf.zeros_initializer(), trainable=False)
            
            for i in range(self.in_length):
                #get g_moduels index
                if i>0:
                    scope.reuse_variables()
                g_counter = 0
                for j in range(self.Ti_sum):
                    if i%self.clockwork_periods[j]==0:
                        g_counter += 1 
                if g_counter == self.Ti_sum: 
                    g_counter = self.hidden_neurons 
                else:  
                    g_counter *= self.Rb 
                
                #at the moment eq1
                tmp_right1 = tf.matmul(X_list[i], tf.slice(self.WI1, [0,0], [-1,g_counter]))  
                tmp_right1 = tf.nn.bias_add(tmp_right1, tf.slice(self.bI1,[0],[g_counter]))
                self.WH1 = tf.multiply(self.WH1, self.traingular_mask)
                tmp_left1 = tf.matmul(self.state1, tf.slice(self.WH1, [0,0], [-1,g_counter]))
                tmp_left1 = tf.nn.bias_add(tmp_left1, tf.slice(self.bH1,[0],[g_counter]))
                tmp_hidden1 = tf.tanh(tf.add(tmp_left1, tmp_right1))
                
                #update hidden layers
                self.state1 = tf.concat(axis=1, values=[tmp_hidden1, tf.slice(self.state1, [0, g_counter], [-1,-1])])      
                
                tmp_right2 = tf.matmul(self.state1, tf.slice(self.WI2, [0,0], [-1,g_counter]))  
                tmp_right2 = tf.nn.bias_add(tmp_right2, tf.slice(self.bI2,[0],[g_counter]))
                self.WH2 = tf.multiply(self.WH2, self.traingular_mask)
                tmp_left2 = tf.matmul(self.state2, tf.slice(self.WH2, [0,0], [-1,g_counter]))
                tmp_left2 = tf.nn.bias_add(tmp_left2, tf.slice(self.bH2,[0],[g_counter]))
                tmp_hidden2 = tf.tanh(tf.add(tmp_left2, tmp_right2))                
                
                self.state2 = tf.concat(axis=1, values=[tmp_hidden2, tf.slice(self.state2, [0, g_counter], [-1,-1])])       
                
            self.final_state = self.state2
            
        return self.final_state

		
class DQNCore(object):
    def __init__(self, observation, num_actions, time_step, start_l_rate, decay_step, decay_rate, cwrnn_hidden_neurons, 
                 cwrnn_Rb, cwrnn_clockwork_periods, gamma, dropout, temp, save_path, test_lr, training=True, loss='MSE'):
        
        self.num_actions = num_actions
        self.gamma = gamma # discount factor for excepted returns 
        self.start_l_rate = start_l_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.global_step = tf.Variable(0, trainable=False)
        if training:
            self.learning_rate = tf.train.exponential_decay(self.start_l_rate, self.global_step, self.decay_step, self.decay_rate, staircase=True) 
        else:
            self.learning_rate = test_lr
        self.dropout = dropout
        self.temp = temp
        self.time_step = time_step
        self.feature_dim = observation.shape[1]
        
        ## below are some parameters about CWRNN 
        self.cwrnn_hidden_neurons = cwrnn_hidden_neurons
        self.cwrnn_Rb = cwrnn_Rb
        if max(cwrnn_clockwork_periods)<=time_step:
            self.cwrnn_clockwork_periods = cwrnn_clockwork_periods
        else:
            raise ValueError("Max clockwork period must be less than time step")
        
        self.save_path1 = save_path+'/training'
        self.save_path2 = save_path+'/trained'
        
        #placeholder for samples replay experience
        self.inputs = tf.placeholder(tf.float32, [1, self.time_step, self.feature_dim])
        self.targets = tf.placeholder(tf.float32, name= 'targets') #y 
        self.actions = tf.placeholder(tf.int32, name= 'actions')
        self.rewards = tf.placeholder(tf.float32, name='rewards')
        self.Q = self._build_CWQNetwork('Qeval', trainable=True) # state Q , main network
        self.next_Q = self._build_CWQNetwork('next_eval',trainable=False) # next state Q , target network
        
        #actions selection corresponding one hot matrix column
        one_hot = tf.one_hot(self.actions, self.num_actions, 1., 0.) ##tf.one_hot(input,one_hot dim,1,0) self.actions為0,1,2
        Qmax = tf.reduce_sum(self.Q * one_hot, axis=1) 

        if loss == 'mse':
            self._loss = tf.reduce_mean(tf.squared_difference(Qmax, self.targets))
        elif loss == 'mse_log':
            epsilon = 1.0e-9
            Qmax = tf.keras.backend.clip(Qmax, epsilon, 1.0 - epsilon)
            self._loss = tf.keras.backend.mean(tf.keras.backend.square(tf.keras.backend.log(self.targets) - tf.keras.backend.log(Qmax)), axis=-1)        
        elif loss == 'mse_sd':
            epsilon = 1.0e-9
            Qmax = tf.keras.backend.clip(Qmax, epsilon, 1.0 - epsilon)
            self._loss = tf.keras.backend.mean(tf.keras.backend.square(self.targets - tf.keras.backend.sqrt(Qmax)), axis=-1)  
        else:
            raise ValueError("action should be elements of ['mse', 'qlike_loss', 'mse_log', 'mse_sd', 'hmse', 'stock_loss']")
            
        self.params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grad_var = optimizer.compute_gradients(loss = self._loss, var_list = self.params, aggregation_method = 2)  
        self._train_op = optimizer.apply_gradients(grad_var, global_step = self.global_step)           
        
        #session
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6)))
        self.sess.run(tf.global_variables_initializer())
        
    def init(self):
        self.state_step = deque(maxlen=self.time_step) 
        self.next_state_step = deque(maxlen=self.time_step) 
            
    def update_state_step(self, state):
        self.state_step.append(state)  
    
    def update_next_state_step(self, next_state):
        self.next_state_step.append(next_state)  
        
    def _build_CWQNetwork(self, name, trainable):  
        w_init, b_init = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            cw_rnn = ClockworkRNN(input_state=self.inputs, in_length=self.time_step, in_width=self.feature_dim, 
                                  hidden_neurons=self.cwrnn_hidden_neurons, Rb=self.cwrnn_Rb, 
                                  clockwork_periods=self.cwrnn_clockwork_periods, trainable=trainable)
            
            final_state = cw_rnn.inference()
            q_network = tf.layers.dense(final_state, self.num_actions, None, kernel_initializer=w_init, bias_initializer=b_init, trainable=trainable, name='output_layer')
            return q_network

    def update_nextQ_network(self):
        next_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='next_eval')
        Q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Qeval')
        #get the min length of zip between them
        self.sess.run([tf.assign(n,q) for n,q in zip(next_params, Q_params)])
    
    def Incremental_Methods(self, action, reward, done): 
        state = np.asarray([self.state_step[i] for i in range(len(self.state_step))])
        next_state = np.asarray([self.next_state_step[i] for i in range(len(self.next_state_step))])
        
        ## Double DQN
        next_Q_main = self.sess.run(self.Q, feed_dict={self.inputs:next_state.reshape(1, self.time_step, self.feature_dim)})
        max_action = np.argmax(next_Q_main[0])
        next_Q_target = self.sess.run(self.next_Q, feed_dict={self.inputs:next_state.reshape(1, self.time_step, self.feature_dim)})
        next_Q = next_Q_target[0][max_action]
        
        #done mask True 1 False 0
        mask = np.array(done).astype('float')
        target = mask * reward + (1 - mask) * (reward + self.gamma * next_Q)

        #op gradient descent step 
        _ , loss = self.sess.run([self._train_op, self._loss], feed_dict={self.inputs:state.reshape(1, self.time_step, self.feature_dim), self.actions:action, self.targets:target})  ##訓練權重
        return loss
    
    def boltzmann_policy_np(self):
        if len(self.state_step) >= self.time_step:  
            state = np.asarray([self.state_step[i][0] for i in range(len(self.state_step))])   
            Q = self.sess.run(self.Q, feed_dict={self.inputs:state.reshape(1, self.time_step, self.feature_dim)})
            Q_probs = self._softmax(Q[0]/self.temp)
            action_value = np.random.choice(Q_probs, p=Q_probs)
            action = np.argmax(Q_probs==action_value)
        else:
            action = 2
            Q_probs = np.array([0,0,0])
        return action, Q_probs    
    
    def greedy_policy(self): 
        if len(self.state_step) >= self.time_step:  
            state = np.asarray([self.state_step[i][0] for i in range(len(self.state_step))])   
            action_value = self.sess.run(self.Q, feed_dict={self.inputs:state.reshape(1, self.time_step, self.feature_dim)})
            action = np.argmax(action_value, axis=1)[0]
        else:
            action = 2
            action_value=np.array([0,0,0])
        return action, action_value
        
    def save_training_model(self): 
        if not os.path.isdir(self.save_path1):
            os.makedirs(self.save_path1)
            
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, self.save_path1+'/save_model.ckpt')
        
        #print('training model save successfully!')
        
    def save_trained_model(self): 
        if not os.path.isdir(self.save_path2):
            os.makedirs(self.save_path2)
            
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, self.save_path2+'/save_model.ckpt')
        
        print('trained model save successfully!')
        
    def load_training_model(self): 
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.save_path1+'/save_model.ckpt')
        
        #print('training model load successfully!')
        
    def load_trained_model(self): 
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.save_path2+'/save_model.ckpt')
        
        #print('trained model load successfully!')
        
    def close_session(self):
        self.sess.close()
        
        print('Close session!')