# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        ### Question 1: no need to change this for now ####

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ##Removed newScaredTimes and newShostStates since it is not used

        #getting the current score here
        #(this was previously returned in the 0/4 test cases)
        currentScore = successorGameState.getScore()

        #use manhattandDistsance to get the closest food               
        foodList = newFood.asList() #needed this as a list        
        if foodList: #check if there is still food                      
            minFoodDistance = 89888888888898
            #this will iterate through the whole foodList, one by one
            for food in foodList:
                manhattanD = manhattanDistance(newPos, food)
                if manhattanD < minFoodDistance: minFoodDistance = manhattanD
            currentScore = currentScore + 10 / minFoodDistance #add reward                       
        return currentScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    #### I don't need to change this for question 1. ####
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    #Still not sure why I had to re-define this function here again
    #The function is defined inside of the getAtction function
    #The issue might be scope, so that I can use it in the getMinValue and gitMaxValue
    def minimax(self, agentIndex, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth == 0: return self.evaluationFunction(gameState) 
        if agentIndex == 0: return self.getMaxValue(agentIndex, depth, gameState, gameState.getNumAgents())
        else: return self.getMinValue(agentIndex, depth, gameState, gameState.getNumAgents())

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth and self.evaluationFunction.
        """
        
        def minimax(agentIndex, depth, gameState):
            #checkinh here if the game reached the terminal state or max depth
            if gameState.isWin() or gameState.isLose() or depth == 0: return self.evaluationFunction(gameState) 
  
            #calling getMaxValue() if its pacman turn      
            #calling getMinValue() if its ghost turn
            if agentIndex == 0: return self.getMaxValue(agentIndex, depth, gameState, gameState.getNumAgents()) 
            else: return self.getMinValue(agentIndex, depth, gameState, gameState.getNumAgents())  

        #track the best score and action possible
        bestAction = None
        bestScore = -8888898889883
        
        #Iterating through all the actions that are legal to pacman
        for action in gameState.getLegalActions(0):
            score = minimax(1, self.depth, gameState.generateSuccessor(0, action)) #get the minimax score
            if score > bestScore: #if better score was found
                bestScore = score
                bestAction = action
        return bestAction

    def getMaxValue(self, agentIndex, depth, gameState, numAgents): #helper function here
        bestScore = -8888898889883
        #Iterate over all legal actions for Pacman
        for action in gameState.getLegalActions(agentIndex):
            score = self.minimax(1, depth, gameState.generateSuccessor(agentIndex, action))
            bestScore = max(bestScore, score) #update the best score to the new one
        return bestScore

    def getMinValue(self, agentIndex, depth, gameState, numAgents): #helper function here
        bestScore = 8888898889883
        #Iterate over all legal actions for ghost
        for action in gameState.getLegalActions(agentIndex):
            #if next agent is Pacman
            if ((agentIndex + 1) % numAgents) == 0: score = self.minimax((agentIndex + 1) % numAgents, depth - 1, gameState.generateSuccessor(agentIndex, action))
            else: score = self.minimax((agentIndex + 1) % numAgents, depth, gameState.generateSuccessor(agentIndex, action))
            bestScore = min(bestScore, score) #update the best score to the new one
        return bestScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alphaBeta(state, depth, agentIndex, alpha, beta):
            #checkinh here if the game reached the terminal state or max depth
            if depth == self.depth * state.getNumAgents() or state.isWin() or state.isLose(): return self.evaluationFunction(state)
            
            #calling getMaxValue() if its pacman turn      
            #calling getMinValue() if its ghost turn
            if agentIndex == 0: return getMaxValue(state, depth, agentIndex, alpha, beta)
            else: return getMinValue(state, depth, agentIndex, alpha, beta)
        
        def getMaxValue(state, depth, agentIndex, alpha, beta): #helper function here
            bestValue = -8888898889883
            bestAction = None
            #Iterate over all legal actions for pacman
            for action in state.getLegalActions(agentIndex):
                value = alphaBeta(state.generateSuccessor(agentIndex, action), depth + 1, (agentIndex + 1) % state.getNumAgents(), alpha, beta)
                
                if type(value) is tuple: value = value[0] #get actual value not tuple
                if value > bestValue: #update bestAction and bestValue if better
                    bestValue = value
                    bestAction = action
                if bestValue > beta: return bestValue #alpha beta pruning
                alpha = max(alpha, bestValue)

            if depth == 0: return bestAction #if root, return bestAction
            return bestValue
        
        def getMinValue(state, depth, agentIndex, alpha, beta): #helper function here
            bestValue = 8888898889883
            #Iterate over all legal actions for ghost
            for action in state.getLegalActions(agentIndex):
                value = alphaBeta(state.generateSuccessor(agentIndex, action), depth + 1, (agentIndex + 1) % state.getNumAgents(), alpha, beta)
                
                if type(value) is tuple: value = value[0] #get actual value not tuple
                if value < bestValue: bestValue = value #update bestValue if better
                if bestValue < alpha: return bestValue  #alpha beta pruning
                beta = min(beta, bestValue)
            return bestValue

        return alphaBeta(gameState, 0, 0, -9998898889883, 9998898889883)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        
        def expectimax(state, depth, agentIndex):
            #checkinh here if the game reached the terminal state or max depth
            if depth == self.depth * state.getNumAgents() or state.isWin() or state.isLose(): return self.evaluationFunction(state)
            
            #calling getMaxValue() if its pacman turn      
            #calling getExpectedValue() if its ghost turn
            if agentIndex == 0: return getMaxValue(state, depth, agentIndex)
            else: return getExpectedValue(state, depth, agentIndex)
        
        def getMaxValue(state, depth, agentIndex): #helper function here
            bestValue = -9998898889883
            bestAction = None
            
            #Iterate over all legal actions for pacman
            for action in state.getLegalActions(agentIndex):
                value = expectimax(state.generateSuccessor(agentIndex, action), depth + 1, (agentIndex + 1) % state.getNumAgents())
                
                if type(value) is tuple: value = value[0] #get actual value not tuple
                if value > bestValue: #update bestValue if better
                    bestValue = value
                    bestAction = action

            if depth == 0: return bestAction #if root, return bestAction
            return bestValue
        
        def getExpectedValue(state, depth, agentIndex): #helper function here
            totalValue = 0
            legalActions = state.getLegalActions(agentIndex)
            
            if not legalActions: return self.evaluationFunction(state) #if no legal move
            
            #Iterate over all legal actions for ghost
            for action in legalActions:
                value = expectimax(state.generateSuccessor(agentIndex, action), depth + 1, (agentIndex + 1) % state.getNumAgents())
                
                if type(value) is tuple: value = value[0] #get actual value not tuple
                totalValue = totalValue + value * 1 / len(legalActions) #total in function of probabilty of the action happening
            return totalValue

        return expectimax(gameState, 0, 0)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    
    #defining variables that will be needed
    result = None
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()

    #Manhattan distance to get the closest food               
    foodList = food.asList()  #needed this as a list
    if foodList:  #check if there is still food                      
        minFoodDistance = 89888888889
        #Iterate through the whole foodList, one by one
        for foodPosition in foodList:
            manhattanD = manhattanDistance(currentGameState.getPacmanPosition(), foodPosition)
            if manhattanD < minFoodDistance: minFoodDistance = manhattanD
    else: minFoodDistance = 1  #quick fix, crashes without this line
    
    #get the distance of the closest ghost
    ghostDistances = []
    for ghostState in currentGameState.getGhostStates(): #iterating through the ghost states
        distance = manhattanDistance(currentGameState.getPacmanPosition(), ghostState.getPosition())
        if ghostState.scaredTimer > 0: ghostDistances.append(-distance)  #if ghost scared
        else: ghostDistances.append(distance)
    
    if ghostDistances: minGhostDistance = min(ghostDistances)
    else: minGhostDistance = 1  #quick fix, crashes without this line
    
    numCapsulesLeft = len(capsules)  #num of capsules left
    numFoodLeft = len(foodList)  #num of food pellets left
    
    #weigths need to calculate the result
    weightScore = 1
    weightFoodDistance = 10
    weightGhostDistance = 5
    weightCapsulesLeft = -100
    weightFoodLeft = -2

    result = (weightScore * currentScore + weightFoodDistance * (1.0 / minFoodDistance) + weightGhostDistance * minGhostDistance + weightCapsulesLeft * numCapsulesLeft + weightFoodLeft * numFoodLeft)
    return result
    
# Abbreviation
better = betterEvaluationFunction
