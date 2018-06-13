# -*- coding: utf-8 -*-
""" Cartpole environment wrapper for Reinforcement Learning

Details:
    File name:          cartpole_wrapper.py
    Date created:       19 March 2018
    Date last modified: 26 March 2018
    Python Version:     3.4

Description:
    Implementation of a wrapper for the CartPole environment as presented in
    https://gym.openai.com/envs/CartPole-v1/

Related files:
    wrapper.py
"""

from wrapper import Wrapper
import math
import pandas as pd


class CartPoleWrapperDiscrete(Wrapper):
    """ This is a wrapper for the CartPole environment as described here:
    https://gym.openai.com/envs/CartPole-v1/

    The discretisation is based on the approach taken by Víctor Mayoral Vilches,
    as explained here:
    https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4

    The following parameters should remain fixed:
    self._actions   Are defined by the environment
    self._pos_lim   Also defined by environment: space in which cart can move
    self._ang_lim   Also defined by environment: maximum allowed angle for pole

    The following parameters can be changed in parameter tuning:
    self._penalty       Used to penalise state-actions that either cause the
                        cart to move out of the frame or the pole to tip too far
    """

    # Some environment-specific parameters
    _actions = [0, 1]   # left (0) or right (1)
    _pos_lim = 2.4      # maximum lateral position
    _ang_lim = 12 * 2 * math.pi / 360  # maximum angle (in radians)

    def __init__(self):
        super().__init__(env_name='CartPole-v1', actions=self._actions)
        self._penalty = -20  # penalty for tipping pole too far or cart
        # running out of frame

        # Define the discretisation of the state vector
        # First define the value ranges
        pos_lim = 2.4  # maximum lateral position
        ang_lim = 12 * 2 * math.pi / 360  # maximum angle (in radians)
        velo_lim = 1  # maximum lateral velocity
        ang_velo_lim = 3.5  # maximum angular velocity

        # Then define the numbers of bins
        n_pos_bins = 10
        n_velo_bins = 10
        n_ang_bins = 10
        n_ang_velo_bins = 10

        # Create the bins
        cart_pos_bins = pd.cut([-pos_lim, pos_lim],
                               bins=n_pos_bins, retbins=True)[1][1:-1]
        cart_velo_bins = pd.cut([-velo_lim, velo_lim],
                                bins=n_velo_bins, retbins=True)[1][1:-1]
        pole_ang_bins = pd.cut([-ang_lim, ang_lim],
                               bins=n_ang_bins, retbins=True)[1][1:-1]
        pole_ang_velo_bins = pd.cut([-ang_velo_lim, ang_velo_lim],
                                    bins=n_ang_velo_bins, retbins=True)[1][1:-1]

        # Set self._bins variable
        self._bins = [cart_pos_bins,
                      cart_velo_bins,
                      pole_ang_bins,
                      pole_ang_velo_bins]

    def get_bins(self):
        """ Returns a list of lists, such that for a state vector (x0, ..., xn),
        the zeroth element of the list contains the list of bins for variable
        x1, the first element of the list contains the list of bins for variable
        x2, and so on.
        """
        return self._bins

    def episode_over(self):
        """ Checks if the episode can be terminated because the maximum number
         of steps (200) has been taken """
        return True if self._number_of_steps >= 200 else False

    def solved(self, rewards):
        """ Checks if environment is solved according to the solve requirements
        specified on https://gym.openai.com/envs/CartPole-v0/, which is:
            'CartPole-v0 defines "solving" as getting average reward of 195.0
            over 100 consecutive trials.'

        :param rewards: list of the episode rewards obtained in the current run
        """
        if len(rewards) < 100:
            return False
        return True if float(sum(rewards[-100:])) / 100 >= 195 else False

    def penalty(self):
        return self._penalty
