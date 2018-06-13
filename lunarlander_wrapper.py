# -*- coding: utf-8 -*-
""" Lunar Lander environment wrapper for Reinforcement Learning

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


class LunarLanderWrapper(Wrapper):
    """ Environment wrapper. Contains only one function of our won to plot the 
    rewards druing the learning process.
    """
    
    _actions = [0, 1, 2, 3]  # Nothing, left, main , right
    mean_rewards, mean_100 = [],[]
    def __init__(self):
        super().__init__(env_name='LunarLander-v2', actions=self._actions)
        
        # Create the bins
        bins = 10
        pos_x = pd.cut([-1, 1], bins=bins, retbins=True)[1][1:-1]
        pos_y = pd.cut([-1, 1], bins=bins, retbins=True)[1][1:-1]
        vel_x = pd.cut([-1, 1], bins=bins, retbins=True)[1][1:-1]                       
        vel_y = pd.cut([-1, 1], bins=bins, retbins=True)[1][1:-1] 
        angle = pd.cut([0, 2*np.pi], bins=bins, retbins=True)[1][1:-1] 
        vel_angle = pd.cut([-1, 1], bins=bins, retbins=True)[1][1:-1] 
        poot1 = np.array([True, False]) 
        poot2 = np.array([True, False])
        # Set self._bins variable
        self._bins = [pos_x, pos_y, vel_x, vel_y, angle, vel_angle, poot1, poot2]
        # Set figure elements
        self.mean_rewards = []
        self.past100_rewards = []
        fig, ax = plt.subplots(figsize=(14,9))



    def get_bins(self):
        """
        Returns a list of lists, such that for a state vector (x0, ..., xn),
        the zeroth element of the list contains the list of bins for variable
        x1, the first element of the list contains the list of bins for variable
        x2, and so on.
        """
        return self._bins




    def plot(self, rewards):
        """
        Plot the results of every episode for one run. The plot shows the
        reward, the mean over all rewards and the mean of the past 100 rewards. 
        """
        # Plot the data        
        self.mean_rewards.append(np.mean(rewards))
        self.past100_rewards.append(np.mean(rewards[-100:]))
    
        plt.clf()
        plt.plot(rewards)
        plt.plot(self.mean_rewards)
        plt.plot(self.past100_rewards)
        plt.legend(['reward episode', 'mean', '100 average'], fontsize=30)
        plt.savefig('frame.png')
        plt.xlabel('Episodes', fontsize = 30)
        plt.ylabel('Rewards', fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.savefig('frame.png')
        plt.pause(0.05)



    def solved(self, rewards):
        """
        :param rewards: a list of rewards received in current run
        :return: a Boolean indicating whether or not the environment is solved
        """
        
        self.plot(rewards)
        
        if (len(rewards) >= 100) and (sum(1 for r in rewards if r >= 200) >= 10):
            return True
        return False
