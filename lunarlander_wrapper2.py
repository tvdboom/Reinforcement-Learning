# -*- coding: utf-8 -*-
""" Lunar Lander environment wrapper for Reinforcement Learning PA - Spring 2018

Details:
    File name:          lunarlander_wrapper.py
    Date created:       28 March 2018
    Python Version:     3.4

Description:
    Implementation of a wrapper for the Lunar Lander environment as presented in
    https://gym.openai.com/envs/LunarLander-v2/

Related files:
    wrapper.py
"""

from wrapper import Wrapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mean_rewards = []
past100_rewards = []

class LunarLanderWrapper(Wrapper):
    """
    Environemnt wrapper. Contains only one function of our own to plot the
    rewards during the learning process. This function is called every episode.
    """
    
    _actions = [0, 1, 2, 3]  # Nothing, left, main , right
    mean_rewards, mean_100 = [],[]
    def __init__(self):
        super().__init__(env_name='LunarLander-v2', actions=self._actions)
        
        # Initialize plot variables
        self.mean_rewards = []
        self.past100_rewards = []
        
        # Initialize figure
        fig, ax = plt.subplots(figsize=(14, 9))
        
        
        
    def plot(self, rewards):
        '''
        Function to plot the data during learning
        '''

        self.mean_rewards.append(np.mean(rewards)) # Get mean all episodes
        self.past100_rewards.append(np.mean(rewards[-100:])) # Get mean last 100 episodes

        plt.clf()
        plt.plot(rewards)
        plt.plot(self.mean_rewards)
        if len(rewards) >= 100:
            plt.plot(self.past100_rewards)
        #plt.legend(['reward episode', 'mean', '100 average'], fontsize=30, loc='lower right')
        plt.xlabel('Episodes', fontsize=45)
        plt.ylabel('Rewards', fontsize=45)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.tight_layout()
        plt.savefig('frame.png')
        


    def solved(self, rewards):
        """
        :param rewards: a list of rewards received in current run
        :return: a Boolean indicating whether or not the environment is solved
        """
        
        self.plot(rewards)
        
        if (len(rewards) >= 100) and (sum(1 for r in rewards if r >= 200) >= 10):
            return True
        return False
