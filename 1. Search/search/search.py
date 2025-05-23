# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """

    #Initialize the frindge as a Stack and the visited nodes as a set 
    stackFringe = util.Stack()
    stackFringe.push((problem.getStartState(),[],0))    
    visited = set()

    while not stackFringe.isEmpty(): #Until stack is empty    
        
        currentNode = stackFringe.pop()
        
        #Check if this is the goal state, return path to goal
        if problem.isGoalState(currentNode[0]):
            return currentNode[1] 
        
        #If the currentNode was not visited
        if not currentNode[0] in visited:
            #Explore currentNode successors
            for successor in problem.getSuccessors(currentNode[0]):
                if not successor[0] in visited:
                    newSuccessor = (successor[0],currentNode[1]+[successor[1]],successor[2])
                    stackFringe.push(newSuccessor)
            visited.add(currentNode[0])
    
    #No solution
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    #Initialize the frindge as a Queue and the visited nodes as a set   
    queueFrindge = util.Queue()
    queueFrindge.push((problem.getStartState(),[],0))    
    visited = set()
    
    while not queueFrindge.isEmpty(): #Until queue is empty   
        
        currentNode = queueFrindge.pop()
        
        #Check if this is the goal state, return path to goal
        if problem.isGoalState(currentNode[0]):
            return currentNode[1]
               
        #If the currentNode was not visited
        if not currentNode[0] in visited:
            #Explore currentNode successors
            for successor in problem.getSuccessors(currentNode[0]):
                if not successor[0] in visited:
                    newSuccessor = (successor[0],currentNode[1]+[successor[1]],successor[2])
                    queueFrindge.push(newSuccessor)
            visited.add(currentNode[0])
    
    #No Solution
    return [] 

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    #Initialize the frindge as a Priority-Queue and the visited nodes as a set     
    pqFringe = util.PriorityQueue()
    pqFringe.push((problem.getStartState(),[],0),0)    
    visited = set()
    
    while not pqFringe.isEmpty(): #Until pq is empty 
        
        currentNode = pqFringe.pop()
    
        #Check if this is the goal state, return path to goal
        if problem.isGoalState(currentNode[0]):
            return currentNode[1]
          
        #If the currentNode was not visited  
        if not currentNode[0] in visited:
            #Explore currentNode successors
            for successor in problem.getSuccessors(currentNode[0]):
                if not successor[0] in visited:
                    newSuccessor = (successor[0],currentNode[1]+[successor[1]],currentNode[2]+successor[2])
                    pqFringe.push(newSuccessor, currentNode[2]+successor[2])
            visited.add(currentNode[0])

    #No solution
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    #Initialize the frindge as a Priority-Queue and the visited nodes as a set     
    pqFringe = util.PriorityQueue()
    pqFringe.push((problem.getStartState(),[],0),0)    
    visited = set()
    
    while not pqFringe.isEmpty(): #Until pq is empty 
        
        currentNode = pqFringe.pop()
        
        #Check if this is the goal state, return path to goal
        if problem.isGoalState(currentNode[0]):
            return currentNode[1]
        
        #If the currentNode was not visited 
        if not currentNode[0] in visited:
            #Explore currentNode successors
            for successor in problem.getSuccessors(currentNode[0]):
                if not successor[0] in visited:
                    newSuccessor = (successor[0], currentNode[1]+[successor[1]], currentNode[2]+successor[2])
                    totalCost = newSuccessor[2] + heuristic(successor[0],problem)
                    pqFringe.push(newSuccessor,totalCost)
            visited.add(currentNode[0])

    #No solution
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
