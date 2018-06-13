# -*- coding: utf-8 -*-
""" Agent class for Reinforcement Learning

Details:
    File name:          my_agent.py
    Author:             Marco Trueba van den Boom
    Date created:       28 March 2018
    Date last modified: 11 May 2018
    Python Version:     3.4

"""

from base_agent import BaseAgent

import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class MyAgent(BaseAgent):
    """ TODO: add description for this class """

    def __init__(self, n_state=8, n_actions=4, gamma=0.99, epsilon=0.15,
                    epsilon_decay=0.995, *args, **kwargs):
                        
        self._wrapper = None
        super().__init__(*args, **kwargs)
       
        self._actions = self._wrapper.actions()     # allowed actions
        # We don't initialise self._q for all possible states. Alternatively, we
        # simply return 0.0 if the state isn't initialised, as we can initialise
        # it arbitrarily.
        self._q = dict()    # key: (state, action), value: corresponding value
        self._bins = self._wrapper.get_bins()
        
        self.maxlen_history = 60000
        self.epochs = 10
        self.batch_size = 32
        
        self.n_state = n_state # Number of variables in a state
        self.n_actions = n_actions # Number of possible actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay # Decay epsilon over time
        self.history = deque(maxlen=self.maxlen_history)
        self.model = self.create_model()
        self.warmup = True
        self.episode = 1
   
   
    def create_model(self, saved = False):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Create the neural network model in Keras. 8 inputs, one for every statechoolt
        variable and 4 outputs, one for every action variable.
        
        '''
        np.random.seed(2)
        nn = Sequential()
        nn.add(Dense(128, input_dim=self.n_state, activation='relu', kernel_initializer = 'lecun_uniform'))
        nn.add(Dense(256, activation='tanh', kernel_initializer = 'lecun_uniform'))
        nn.add(Dense(self.n_actions, activation='linear', kernel_initializer = 'lecun_uniform'))
        nn.compile(loss='mse', optimizer=Adam(lr=0.0001))
        return nn
        
    def save_model(self):
        ''' 
        DESCRIPTION -----------------------------------------------------------
        
        Saves the model after the game is solved 

        '''
        self.model.save_weights(path_out + 'weights_{0}.hdf5'.format(episode))


    def fit_model(self, verbosity=0):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Calculates the estimated Q value and fits the network on these values. 
        
        '''        
        
        states, targets = [], []
        for state, action, reward, next_state, done in self.history:
            target = reward
            if not done: # In case the environment is solved already
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
                
            target_future = self.model.predict(state)
            target_future[0][action] = target
            
            states.append(state)
            targets.append(target_future)
        
        states = np.squeeze(states)
        targets = np.squeeze(targets)
        
        loss = self.model.fit(states, targets, batch_size = self.batch_size, epochs=self.epochs, verbose=verbosity)
        with open('loss_zonder_landen.txt', 'a') as f:
            f.write(str(loss.history['loss'][-1]) + '\n')
        f.close()
        # Add epsilon decay technique
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
            


    def save_history(self, state, action, reward, next_state, done):
        """
        DESCRIPTION -----------------------------------------------------------

        Adds the new states to the history. If the history is bigger than maxlen_history,
        the first elements are removed from the list.
        """
        self.history.append((state, action, reward, next_state, done))
        if len(self.history) >= self.maxlen_history:
            del self.history[0]
            
            
    def initialise_episode(self):
        """
        Reset the total reward for the episode. Reset the environment through
        the wrapper.

        :return: initial state vector
        """
        self.total_reward = 0
        return self._wrapper.reset()
        
        
    def get_action_value(self, state, action):
        """Return value for (state, action) if this tuple is in self._q.
        Otherwise return 0.

        :param state:   an iterable object representing a state
        :param action:  an iterable object representing an action
        :return:        the value stored in self._q at key (state, action) if
                        that key exists. Otherwise return 0.
        """
        return self._q.get((state, action), 0.0)
        
        
    def val2bin(self, val, bins):
        """
        Returns the index of the bin in which value val falls.

        :param val:     a float or int
        :param bins:    the values that separate the bins
        :return:        a bin index
        """
        return np.digitize(x=[val], bins=bins)[0]


    def digitise_state(self, state):
        """
        For each element of the state vector, determine to which bin it
        belongs. Turn the resulting vector into a string for the hashing of
        the dictionary.

        :param state:   a state vector of length n
        :return:        a string that concatenates the n bin indices obtained
                        from discretising state. indices separated by commas
        """
        return np.array([[self.val2bin(v, bins)
                    for (v, bins) in zip(state, self._bins)]])       


    def select_action(self, state, *args):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Select action for next state. Uses epsilon-greedy technique.
        
        '''   

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions) # Random next state
        else:
            return np.argmax(self.model.predict(state)) # Networks prediction



    def train(self):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Trains the algorithm. For each run there is a warm up period where for
        1000 episodes random actions are taken. This data is added to the history
        and afterwards the network is fitted based on this data to have a already 
        trained network at the start of the episode. 
        For every state an action is chosen based on the predictions of the model.
        Every 10 episodes the network is refitted on the new data. Every episode the 
        states and rewards are saved in the history.
        
        '''   
        # Initialise the episode and environment
        state = self.digitise_state(self.initialise_episode())
        done = False
        local_history = []
        print ('Epsilon: %.4f' %self.epsilon)

        if len(self.history) > 0: self.warmup=False
        # Warm up: do 1000 episodes without training
        if self.warmup == True:
            print('Warm up')
            for warmup_time in range(1000):
                
                action = self.select_action(state)
                next_state, reward, done, _ = self._wrapper.step(action)
                self.save_history(state, action, reward, next_state, done)
                # Extra dimension is needed for correct implementation in
                # the neural network
                state = np.array([next_state])
                if done:
                    if warmup_time == 1000: 
                        self.fit_model()
                        self.warmup = False
                    

        # Start training
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _ = self._wrapper.step(action)
            self.total_reward += reward 
            
            print('\rAction selected: %i   Total reward: %.2f ' %(action, self.total_reward), end='  ')

            #local_history.append((state, action, reward, next_state, done))
            self.save_history(state, action, reward, next_state, done)
            
            # Extra dimension is needed for correct implementation in
            # the neural network
            state = np.array([next_state])
            
            if done:
                # Fit every 10 episodes
                if self.episode%10 == 0: 
                    self.fit_model()
                    print('Fitted')
                self.episode += 1
                break
        
        return self.total_reward















