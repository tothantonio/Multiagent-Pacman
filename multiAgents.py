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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        food_list = newFood.asList()
        if food_list:
            min_food_distance = min(util.manhattanDistance(newPos, food) for food in food_list)
            score += 10.0 / (min_food_distance + 1)
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDistance = util.manhattanDistance(newPos, ghostPos)
            scared_time = newScaredTimes[i]

            if scared_time == 0:
                if ghostDistance < 3:
                    score -= 500.0 / (ghostDistance + 1)
            else:
                if ghostDistance < scared_time:
                    score += 20.0 / (ghostDistance + 1)
        if action == Directions.STOP:
            score -= 10.0
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        _, action = self.max_value(gameState, 0, 0)
        return action

    def minmax(self, gameState, agentIdx, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIdx == 0 and depth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIdx == 0:
            return self.max_value(gameState, 0, depth)
        else:
            return self.min_value(gameState, agentIdx, depth)

    def max_value(self, gameState, agentIdx, depth):
        max_val = float('-inf')
        best_action = Directions.STOP
        for action in gameState.getLegalActions(agentIdx):

            succ = gameState.generateSuccessor(agentIdx, action)
            newDepth = depth + 1

            score = self.minmax(succ, 1, newDepth)

            if score > max_val:
                max_val = score
                best_action = action
        if depth == 0:
            return max_val, best_action
        else:
            return max_val

    def min_value(self, gameState, agentIdx, depth):
        min_val = float('inf')
        numAgents = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIdx):

            succ = gameState.generateSuccessor(agentIdx, action)
            nextAgent = agentIdx + 1

            if nextAgent < numAgents:
                score = self.minmax(succ, nextAgent, depth)
            else:
                score = self.minmax(succ, 0, depth)
            min_val = min(min_val, score)
        return min_val

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, action = self.max_value(gameState, 0, 0, float('-inf'), float('inf'))
        return action

    def alpha_beta(self, gameState, agentIdx, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIdx == 0 and depth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIdx == 0:
            return self.max_value(gameState, agentIdx, depth, alpha, beta)
        else:
            return self.min_value(gameState, agentIdx, depth, alpha, beta)

    def max_value(self, gameState, agentIdx, depth, alpha, beta):
        max_val = float('-inf')
        best_action = Directions.STOP
        for action in gameState.getLegalActions(agentIdx):
            succ = gameState.generateSuccessor(agentIdx, action)
            newDepth = depth + 1

            score = self.alpha_beta(succ, 1, newDepth, alpha, beta)

            if score > max_val:
                max_val = score
                best_action = action

            if max_val > beta:
                if depth == 0:
                    return max_val, best_action
                else:
                    return max_val

            alpha = max(alpha, score)

        if depth == 0:
            return max_val, best_action
        else:
            return max_val

    def min_value(self, gameState, agentIdx, depth, alpha, beta):
        min_val = float('inf')
        numAgents = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIdx):
            succ = gameState.generateSuccessor(agentIdx, action)
            nextAgent = agentIdx + 1

            if nextAgent < numAgents:
                score = self.alpha_beta(succ, nextAgent, depth, alpha, beta)
            else:
                score = self.alpha_beta(succ, 0, depth, alpha, beta)

            min_val = min(min_val, score)

            if min_val < alpha:
                return min_val

            beta = min(beta, score)
        return min_val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
