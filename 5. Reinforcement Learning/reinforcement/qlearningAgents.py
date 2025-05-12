# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    # *********************
    #    Question 6 
    # *********************
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()  # Initialize Q-values as a Counter (dictionary with default 0)

    # *********************
    #    Question 6 
    # *********************
    def getQValue(self, state, action):
        return self.q_values[(state, action)]

    # *********************
    #    Question 6 
    # *********************
    def computeValueFromQValues(self, state):
        if not self.getLegalActions(state): return 0.0 #if no action, return 0.0
        #return max q value
        return max([self.getQValue(state, action) for action in self.getLegalActions(state)])

    # *********************
    #    Question 6 
    # *********************
    def computeActionFromQValues(self, state):
        if not self.getLegalActions(state): return None #if no legal action, do nothing

        bestActions = []
        maxQ = -9999999
        
        for action in self.getLegalActions(state): #for all action that are legal, do
            q = self.getQValue(state, action)
            if q > maxQ: #update by swaping them
                maxQ = q
                bestActions = [action]
            elif q == maxQ: bestActions.append(action)

        return random.choice(bestActions)

    # *********************
    #    Question 7 
    # *********************
    def getAction(self, state):
        if not self.getLegalActions(state):  return None #if no legal actions, so nothing

        if util.flipCoin(self.epsilon): return random.choice(self.getLegalActions(state))
        else: return self.computeActionFromQValues(state)


    # *********************
    #    Question 6 
    # *********************
    def update(self, state, action, nextState, reward): 
        self.q_values[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    # *********************
    #    Question 10
    # *********************
    def getQValue(self, state, action):
        return sum([self.weights[feature] * value for feature, value in self.featExtractor.getFeatures(state, action).items()])

    # *********************
    #    Question 10 
    # *********************
    def update(self, state, action, nextState, reward):
        diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for feature, value in self.featExtractor.getFeatures(state, action).items():
            self.weights[feature] += self.alpha * diff * value

    def final(self, state):
        QLearningAgent.final(self, state)
        if self.episodesSoFar == self.numTraining: print("Final weights after training:" + self.weights) # Print out the weights
