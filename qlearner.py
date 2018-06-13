# -*- coding: utf-8 -*-
""" Q-learning Agent class for Reinforcement Learning

Details:
    File name:          qlearner.py
    Date created:       19 March 2018
    Date last modified: 26 March 2018
    Python Version:     3.4

Description:
    Implementation of a Q-learning class that can deal with discretisation of
    continuous state spaces.

Related files:
    base_agent.py
"""

import numpy as np
from base_agent import BaseAgent


class QLearner(BaseAgent):
    """ A q-learning class based on Víctor Mayoral Vilches' Q-learning algorithm
    from https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4
    and on the Q-learning (off-policy TD control) algorithm as described in
        Sutton and Barto
        Reinforcement Learning: An Introduction, Chapter 6.5
        2nd edition, Online Draft, January 1 2018 version, retrieved from
        http://incompleteideas.net/book/the-book-2nd.html
    """

    def __init__(self, gamma=0.9, *args, **kwargs):
        self._wrapper = None
        super().__init__(*args, **kwargs)
        self._gamma = gamma                         # discount factor
        self._actions = self._wrapper.actions()     # allowed actions
        # We don't initialise self._q for all possible states. Alternatively, we
        # simply return 0.0 if the state isn't initialised, as we can initialise
        # it arbitrarily.
        self._q = dict()    # key: (state, action), value: corresponding value
        self._bins = self._wrapper.get_bins()

    def initialise_episode(self):
        """
        Reset the total reward for the episode. Reset the environment through
        the wrapper.

        :return: initial state vector
        """
        self._total_reward = 0
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

    def learnQ(self, state, action, reward, maxqnew):
        """
        Apply Q-learning rule:
            Q(s, a) += alpha * (reward(s, a) + max(Q(s') - Q(s, a))

        :param state:   an iterable object representing a state
        :param action:  an iterable object representing an action
        :param reward:  a float or int representing the reward received for
                        taking action action in state state
        :param maxqnew: value corresponding to the best action as seen from
                        the new state
        """
        old_value = self._q.get((state, action), None)
        new_value = reward + self._gamma * maxqnew
        if old_value is None:
            self._q[(state, action)] = reward
        else:
            self._q[(state, action)] = old_value + \
                                       self._alpha * (new_value - old_value)

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
        return ",".join([str(self.val2bin(v, bins))
                         for (v, bins) in zip(state, self._bins)])

    def select_action(self, state):
        """
        Get the state-action values and use them as input for selecting a state
        in a epsilon-greedy fashion.

        :param state:   a state vector
        :return:        an action index
        """
        q = [self.get_action_value(state, a) for a in self._actions]
        a_id = self.epsilon_greedy(q)
        return self._actions[a_id]

    def train(self):
        """
        Run one episode of Q-learning.

        :return: Total reward for this episode
        """
        # Initialise the episode and environment
        state = self.digitise_state(self.initialise_episode())

        # Main loop of training session
        while True:
            # Select an action, take it and observe outcome
            a_id = self.select_action(state)  # index of the action
            action = self._actions[a_id]      # the corresponding action
            cont_new_state, reward, done, _ = self._wrapper.step(action)
            self._total_reward += reward

            # If we reached a terminal state, reward strong penalty for the
            # learning process
            if done:
                reward = self._wrapper.penalty()

            # If we haven't reached the terminal state, but the number of steps
            # runs out, consider this episode done
            if self._wrapper.episode_over():
                done = True

            # Convert continuous state vector to bins
            new_state = self.digitise_state(cont_new_state)

            # Determine the maximum action value
            maxqnew = max([self.get_action_value(new_state, a)
                           for a in self._actions])

            # Learn
            self.learnQ(state, action, reward, maxqnew)

            state = new_state
            if done:
                break

        # At the end of the learning session, return the total reward
        return self._total_reward

