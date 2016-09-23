import numpy as np
import random
import os
import time
import csv
import math
import sys

from scipy.ndimage import imread
import scipy.io as sio
import scipy

import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Flatten, Permute, Lambda, Layer, RepeatVector, merge
from keras.layers import Convolution2D, Convolution3D, MaxPooling2D, AveragePooling2D
from keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Recurrent
from keras.layers import BatchNormalization
from keras.activations import relu
from keras import activations, initializations, regularizers
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam

from keras import backend as K
from keras.engine import Layer, InputSpec

import matplotlib.pyplot as plt

import gym

from policy import epsilon_greedy, softmax_policy

render = True # Cannot render in SSH connection

seq_length = 512
epochs = 200
print_epoch = 10

env = gym.make('CartPole-v0')
nb_actions = env.action_space.n
nb_features = env.observation_space.shape[0]

print 'Building models'

def build_model(train = False):
    model = Sequential()
    if train == False:
        model.add(LSTM(50, return_sequences = True, stateful = True, consume_less = 'gpu', batch_input_shape = (1, 1, nb_features)))
        #model.add(LSTM(50, return_sequences = True, stateful = True, consume_less = 'gpu'))
    else:
        model.add(LSTM(50, return_sequences = True, stateful = False, consume_less = 'gpu', batch_input_shape = (1, None, nb_features)))
        #model.add(LSTM(50, return_sequences = True, stateful = False, consume_less = 'gpu'))
    model.add(TimeDistributed(Dense(nb_actions, activation = 'softmax')))
    return model

def softmax_policy_search(reward, actions): #(y_true, y_pred)
    selected = K.switch(K.equal(reward, 0), 0, 1) # Assume reward/penalty for 1 action at each timestep
    selected_actions = K.clip(actions, 1e-8, 1 - 1e-8)  * selected # Clip for numerical stability

    # Sum over feature axis then timesteps. Note we use a negative sign to turn maximisation into minimisation for Keras
    axis = range(2, actions.ndim)
    print axis

    selected_actions = K.sum(selected_actions, axis = axis)

    logs = K.log(selected_actions)

    reward_dimshuffle = K.permute_dimensions(K.sum(reward, axis = axis), (1, 0)) # Theano will scan across axis 0
    print reward_dimshuffle.ndim

    def value_func(reward_history, state):
        return reward_history + state

    # Sum from bottom up
    value_dimshuffle, _ = theano.scan(fn = value_func, sequences = reward_dimshuffle, outputs_info = T.zeros_like(reward_dimshuffle[0]), go_backwards = True)

    value = K.permute_dimensions(value_dimshuffle[::-1], (1, 0)) # Theano scan output is upside down so we flip, we also need to undo the permute

    expectation = -K.sum(logs * value, axis = 1) # Keras will mean over batch axis for us
    return expectation

# We perform derivative of the policy and multiply by reward sum outside of Keras.
# Then we just multiply with network derivative by chain rule
def policy_search(policy_derivative, network): #(y_true, y_pred)
    return -K.sum(network * policy_derivative, axis = 1)

model_run = build_model(train = False) # We will use this one to run and generate a series of actions
model_train = build_model(train = True) # We will use this one to train once we know the actions
model_run.set_weights(model_train.get_weights())

print 'Compiling models'

model_run.compile(loss = 'mse', optimizer = 'adam') # Loss and optimizer does not matter as we will not use this model to train
model_train.compile(loss = policy_search, optimizer = 'adam')

print 'Training model'
reward_average = 0
for j in xrange(epochs):

    reward_total = 0

    state_history = []
    reward_history = []
    ln_derivative_history = []

    # Reset game environment
    state = env.reset()

    model_run.reset_states()
    for t in xrange(seq_length):
        #if render == True:
            #env.render()
        state_history += [np.reshape(state, (1, 1, -1))]
        # Run the run model
        action = model_run.predict_on_batch(np.reshape(state, (1, 1, -1)))
        action, ln_derivative = softmax_policy(action)
        action = np.argmax(action)
        state, reward, done, info = env.step(action)
        #reward_onehot = np.zeros((1, 1, nb_actions))
        #reward_onehot[0, 0, action] = reward if reward != 0 else -1e-8
        reward_history += [reward]
        ln_derivative_history += [ln_derivative]
        reward_total += reward

        if done:
            break

    state_history = np.concatenate(state_history, axis = 1)
    reward_history = np.array(reward_history)
    ln_derivative_history = np.concatenate(ln_derivative_history, axis = 1)
    #print ln_derivative_history.shape

    policy_derivative = np.zeros_like(ln_derivative_history)
    for i in xrange(policy_derivative.shape[1]):
        policy_derivative[:, i] = np.sum(reward_history[i:]) * ln_derivative_history[:, i]

    model_train.train_on_batch(state_history, policy_derivative)
    model_run.set_weights(model_train.get_weights())
    reward_average = reward_total * 0.5 + reward_average * 0.5
    if (j + 1) % print_epoch == 0:
        print 'Epoch, ', j, ' ', reward_average

reward_total = 0
state = env.reset()
while True:
    if render == True:
        env.render()
    state = np.reshape(state, (1, 1, -1))
    action = np.argmax(model_run.predict_on_batch(np.reshape(state, (1, 1, -1))))
    state, reward, done, info = env.step(action)
    reward_total += reward
    if done:
        break

print 'Final run through reward', reward_total
