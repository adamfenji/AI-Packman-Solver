# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    # *********************
    #    Question 1
    # *********************
    def runValueIteration(self):
        for i in range(self.iterations): #Run value iteration for the specified number of iterations
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state): newValues[state] = 0
                else: #Calculate the value for this state based on possible actions
                    actionValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
                    newValues[state] = max(actionValues) if actionValues else 0
            self.values = newValues #Update values right here


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    # *********************
    #    Question 1
    # *********************
    def computeQValueFromValues(self, state, action):
        q = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            #Bellman update method here
            q += prob * ((self.mdp.getReward(state, action, nextState)) + self.discount * self.values[nextState])
        return q

    # *********************
    #    Question 1 
    # *********************
    def computeActionFromValues(self, state):
        #No action in terminal states
        if self.mdp.isTerminal(state): return None

        bestAction = None
        maxValue = -9999999999
        for action in self.mdp.getPossibleActions(state): #For all possible actions, until there is nothing to be done
            q = self.computeQValueFromValues(state, action)
            if q > maxValue:
                maxValue = q
                bestAction = action #Swapping them right here

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    # *********************
    #    Question 4
    # *********************
    def runValueIteration(self):
        
        for i in range(self.iterations):
            state = self.mdp.getStates()[i % len(self.mdp.getStates())] #Select state cyclically
            if self.mdp.isTerminal(state):
                continue  #I want to skip terminal states

           #get max q value for the current state
            actionValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
            self.values[state] = max(actionValues) if actionValues else 0



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    # *********************
    #    Question 5 
    # *********************
    def runValueIteration(self):
        
        #get predecessors of all states
        predecessors = {state: set() for state in self.mdp.getStates()}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0: predecessors[nextState].add(state)

        pq = util.PriorityQueue() #this an empty pq

        # For each non-terminal state...
        # get initial priorities and add to the queue
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                maxQ = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                pq.push(state, -abs(self.values[state] - maxQ))

        for iteration in range(self.iterations):
            if pq.isEmpty(): break #empty, then just end this
            state = pq.pop() #get state

            #update the value of the state
            if not self.mdp.isTerminal(state): self.values[state] = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])

            #update the priorities of the predecessors
            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    maxQ = max([self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)])
                    if abs(self.values[p] - maxQ) > self.theta: pq.update(p, -abs(self.values[p] - maxQ))

