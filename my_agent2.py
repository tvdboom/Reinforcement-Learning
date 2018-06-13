# -*- coding: utf-8 -*-
""" Agent class for Reinforcement Learning

Details:
    File name:          my_agent.py
    Author:             Marco Trueba van den Boom
    Date created:       28 March 2018
    Date last modified: 11 May 2018
    Python Version:     3.4

Description:
    This agent combines multiple neural networks (one for every action) to
    learn which action to take under every state. It applies standard
    Q-learning to fit the networks.
        

"""

from base_agent import BaseAgent

import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adamax


def create_model(n_state):
    '''Create neural network model.'''
    
    np.random.seed(2) # Fix seed for reproducibility
    model = Sequential()
    model.add(Dense(128, kernel_initializer='lecun_uniform', input_shape=(n_state,), activation='relu'))
    model.add(Dense(256, kernel_initializer='lecun_uniform', activation='tanh'))
    model.add(Dense(1, kernel_initializer='lecun_uniform', activation='linear'))
    model.compile(loss='mse', optimizer=Adamax())
    return model 
      
      
class MyAgent(BaseAgent):
    '''
    This agent combines multiple neural networks (one for every action) to
    learn which action to take under every state. It applies standard
    Q-learning to fit the networks.
    '''
    
    def __init__(self, n_state=8, n_actions=4, gamma=0.99, epsilon=0.15,
                    epsilon_decay=0.995, verbose=0, *args, **kwargs):
                        
        self._wrapper = None
        super().__init__(*args, **kwargs)
       
        self._actions = self._wrapper.actions()     # allowed actions
        self._q = dict()    # key: (state, action), value: corresponding value
        
        # Parameters
        self.n_state = n_state # Number of variables in a state
        self.n_actions = n_actions # Number of possible actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay # Decay epsilon over time

        self.scaler = StandardScaler() #sklearn scaler
        self.verbose = verbose
        self.solved = 0
        
        # Create multiple neural networks
        self.models = []
        for _ in range(self.n_actions):
            model = create_model(self.n_state)
            self.models.append(model) 

        '''
        At the start, run 1000 random episodes to fit the scaler.
        '''
        
        # Initialize array for saving samples
        states_samples = []

        for n in range(1000):
            state = self.initialise_episode()
            states_samples.append(state)
            done = False
            while not done:
                action = np.random.randint(4) # Select random action
                next_state, reward, done, _ = self._wrapper.step(action)
                states_samples.append(next_state)
                
        states_samples = np.array(states_samples)
        self.scaler.fit(states_samples) # Fit scaler
    
     
    def predict(self, state):
        '''Return array of predictions of the four NNs'''
        
        X = self.scaler.transform(np.atleast_2d(state)) # Scaler needs to 2D array
        return np.array([model.predict(np.array(X), verbose=0)[0] for model in self.models])


    def update(self, state, action, target):
        '''Fit the model'''
        
        # Scale the features subtracting the mean and dividing by the
        # standard deviation. 
        X = self.scaler.transform(np.atleast_2d(state)) 
        self.models[action].fit(np.array(X), np.array([target]),
                                    epochs=1, verbose=self.verbose)


    def select_action(self, state):
        '''Return selected action for a given state using the e-greedy method.'''
        
        if np.random.random() < self.epsilon:
            return random.randrange(self.n_actions) # Random next state
        else:
            return np.argmax(self.predict(state)) # Next state from NN

     
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

  
    def train(self):
        
        '''         
        Train the algorithm using standard Q-learning to fit the model.
        '''

        
        # Initialise the episode and environment
        state = self.initialise_episode()
        done = False

        # Loop until the spaceship crashes or flies outside the allowed range
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _ = self._wrapper.step(action)
            self.total_reward += reward
            
            # Q-learning
            next = self.predict(next_state)
            target = reward + self.gamma * np.max(next)
            self.update(state, action, target) # Fit the model
        
            state = next_state
        
        # Count number of times the environment has been solved
        if self.total_reward > 200:
            self.solved += 1
            
        print("total reward: %.2f   eps: %.3f   solved: %i" %(self.total_reward, self.epsilon, self.solved))                           
        
        # Apply epsilon decay technique
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
        
        return self.total_reward























