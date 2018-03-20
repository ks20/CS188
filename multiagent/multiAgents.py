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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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

        newFood = successorGameState.getFood() #double array
        newFoodList = newFood.asList()

        foodDistances = []
        for foodPos in newFoodList:
            foodDistances += [manhattanDistance(newPos, foodPos)]



        newGhostStates = successorGameState.getGhostStates()
        ghostDistances = []
        for ghostState in newGhostStates:
            ghostDistances += [manhattanDistance(newPos, ghostState.getPosition())]


        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        minScaredTime = min(newScaredTimes)


        "*** YOUR CODE HERE ***"
        total = successorGameState.getScore()
        if len(foodDistances) != 0:
            closestFood = float(min(foodDistances))
            total += (1/closestFood)
        if len(ghostDistances) != 0 and min(ghostDistances) != 0:
            closestGhost = float(min(ghostDistances))
            total -= (1/closestGhost)

        return total + minScaredTime

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
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

    def getAction(self, gameState):
        return self.maxValue(gameState, 0, self.depth)[0]
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """


    def maxValue(self, gameState, agentIndex, currDepth):
        if gameState.isWin() or gameState.isLose() or currDepth == 0:
            return None, self.evaluationFunction(gameState)
        val = -float('inf')
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            newState = gameState.generateSuccessor(agentIndex, action)
            h = (self.minValue(newState, 1, currDepth))[1]
            val = max(val, h)
            if h == val:
                bestAction = action
        return (bestAction, val)

    def minValue(self, gameState, agentIndex, currDepth):
        if gameState.isWin() or gameState.isLose() or currDepth == 0:
            return None, self.evaluationFunction(gameState)
        val = float('inf')
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            newState = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex == (gameState.getNumAgents() - 1)):
                h = (self.maxValue(newState, 0, currDepth-1))[1]
                val = min(val, h)
                if h == val:
                    bestAction = action
            else:
                h = self.minValue(newState, agentIndex+1, currDepth)[1]
                val = min(val, h)
                if h == val:
                    bestAction = action
        return (bestAction, val)

"""
    def max-value(state):
        initiallize v = -inf
        for each successor of state:
            v = max(v,value(successor))
        return v

    def min-value(state):
        initiallize v = +inf
        for each successor of state:
            v = min(v,value(successor))
        return v

    def value(state):
        if state is a terminal state, return that states utility
        if the next agent is max, return max-value(state)
        if the next agent is min, return min-value(state)"""

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        return self.maxValue(gameState, 0, self.depth, float('-inf'), float('inf'))[0]
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """util.raiseNotDefined()"""

    def maxValue(self, gameState, agentIndex, currDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or currDepth == 0:
            return None, self.evaluationFunction(gameState)
        val = -float('inf')
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            newState = gameState.generateSuccessor(agentIndex, action)
            h = (self.minValue(newState, 1, currDepth, alpha, beta))[1]
            val = max(val, h)
            if val > beta:
                return (action, val)
            alpha = max(alpha, val)
            if h == val:
                bestAction = action
        return (bestAction, val)

    def minValue(self, gameState, agentIndex, currDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or currDepth == 0:
            return None, self.evaluationFunction(gameState)
        val = float('inf')
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            newState = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex == (gameState.getNumAgents() - 1)):
                h = (self.maxValue(newState, 0, currDepth-1, alpha, beta))[1]
                val = min(val, h)
                if h == val:
                    bestAction = action
                if val < alpha:
                    return (action, val)
                beta = min(beta, val)
            else:
                h = self.minValue(newState, agentIndex+1, currDepth, alpha, beta)[1]
                val = min(val, h)
                if h == val:
                    bestAction = action
                if val < alpha:
                    return (action, val)
                beta = min(beta, val)
        return (bestAction, val)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        return self.maxValue(gameState, 0, self.depth)[0]
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

    def expectiValue(self, gameState, agentIndex, currDepth):
        if gameState.isLose() or gameState.isWin() or currDepth == 0:
            return (None, float(self.evaluationFunction(gameState)))
        actions = gameState.getLegalActions(agentIndex)
        total = 0.0
        for action in actions:
            newState = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex == (gameState.getNumAgents() - 1)):
                total += float(self.maxValue(newState, 0, currDepth-1)[1])
            else:
                total += float(self.expectiValue(newState, agentIndex+1, currDepth)[1])
        return (None, total/float(len(actions)))

    def maxValue(self, gameState, agentIndex, currDepth):
        if gameState.isWin() or gameState.isLose() or currDepth == 0:
            return (None, float(self.evaluationFunction(gameState)))
        val = -float('inf')
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            newState = gameState.generateSuccessor(agentIndex, action)
            h = (self.expectiValue(newState, 1, currDepth)[1])
            val = max(val, h)
            if h == val:
                bestAction = action
        return (bestAction, val)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <score at the current state + 1/(distance to nearst food) - 2/(distance to nearest ghost) (in order to play conservatively>
    """
    newPos = currentGameState.getPacmanPosition()

    newFood = currentGameState.getFood()  # double array
    newFoodList = newFood.asList()

    foodDistances = []
    for foodPos in newFoodList:
        foodDistances += [manhattanDistance(newPos, foodPos)]

    newGhostStates = currentGameState.getGhostStates()
    ghostDistances = []
    for ghostState in newGhostStates:
        ghostDistances += [manhattanDistance(newPos, ghostState.getPosition())]

    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    minScaredTime = min(newScaredTimes)

    "*** YOUR CODE HERE ***"
    total = currentGameState.getScore()
    if len(foodDistances) != 0:
        closestFood = float(min(foodDistances))
        total += (1 / closestFood)
    if len(ghostDistances) != 0 and min(ghostDistances) != 0:
        closestGhost = float(min(ghostDistances))
        total -= 2*(1 / closestGhost)

    return total + minScaredTime

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

