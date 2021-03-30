from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    """
    return currentGameState.getScore()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    def _scoreFromGhost(gameState):
      score = 0
      for ghost in gameState.getGhostStates():
        disGhost = manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition())
        if ghost.scaredTimer > 0:
          score += pow(max(8 - disGhost, 0), 2)
        else:
          score -= pow(max(7 - disGhost, 0), 2)
      return score

    def _scoreFromFood(gameState):
      disFood = []
      for food in gameState.getFood().asList():
        disFood.append(1.0/manhattanDistance(gameState.getPacmanPosition(), food))
      if len(disFood)>0:
        return max(disFood)
      else:
        return 0

    def _scoreFromCapsules(gameState):
      score = []
      for Cap in gameState.getCapsules():
        score.append(50.0/manhattanDistance(gameState.getPacmanPosition(), Cap))
      if len(score) > 0:
        return max(score)
      else:
        return 0

    def _suicide(gameState):
      score = 0
      disGhost = 1e6
      for ghost in gameState.getGhostStates():
        disGhost = min(manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition()), disGhost)
      score -= pow(disGhost, 2)
      if gameState.isLose():
        score = 1e6
      return score

    score = currentGameState.getScore()
    scoreGhosts = _scoreFromGhost(currentGameState)
    scoreFood = _scoreFromFood(currentGameState)
    scoreCapsules = _scoreFromCapsules(currentGameState)
    return score + scoreGhosts + scoreFood + scoreCapsules


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
    Your minimax agent (question 3)
    """

    def max_value(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if depth == 0:
            return self.evaluationFunction(gameState), None

        best_value = -999999999
        best_action = None
        actions = gameState.getLegalActions(index)
        for action in actions:
            next_state = gameState.generateChild(index, action)
            if index == self.index:
                value, _ = self.min_value(next_state, index+1, depth)
            else:
                value, _ = self.max_value(next_state, index+1, depth)

            if value > best_value:
                best_value = value
                best_action = action
        return best_value, best_action

    def min_value(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        best_value = 999999999
        best_action = None
        actions = gameState.getLegalActions(index)
        for action in actions:
            next_state = gameState.generateChild(index, action)
            if index == gameState.getNumAgents() - 1:
                value, _ = self.max_value(next_state, 0, depth-1)
            else:
                value, _ = self.min_value(next_state, index+1, depth)

            if value < best_value:
                best_value = value
                best_action = action
        return best_value, best_action

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        we assume ghosts act in turn after the pacman takes an action
        so your minimax tree will have multiple min layers (one for each ghost)
        for every max layer

        gameState.generateChild(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state

        self.evaluationFunction(state)
        Returns pacman SCORE in current state (useful to evaluate leaf nodes)

        self.depth
        limits your minimax tree depth (note that depth increases one means
        the pacman and all ghosts has already decide their actions)
        """
        _, best_action = self.max_value(gameState, 0, self.depth)
        print()
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_value(self, gameState, index, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if depth == 0:
            return self.evaluationFunction(gameState), None

        best_value = -999999999
        best_action = None
        actions = gameState.getLegalActions(index)
        for action in actions:
            next_state = gameState.generateChild(index, action)
            if index == self.index:
                value, _ = self.min_value(next_state, index+1, depth, alpha, beta)
            else:
                value, _ = self.max_value(next_state, index+1, depth, alpha, beta)

            if value > best_value:
                best_value = value
                best_action = action

            if value >= beta:
                return value, action

            alpha = max(alpha, value)

        return best_value, best_action

    def min_value(self, gameState, index, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        best_value = 999999999
        best_action = None
        actions = gameState.getLegalActions(index)
        for action in actions:
            next_state = gameState.generateChild(index, action)
            if index == gameState.getNumAgents() - 1:
                # here
                value, _ = self.max_value(next_state, 0, depth-1, alpha, beta)
            else:
                value, _ = self.min_value(next_state, index+1, depth, alpha, beta)

            if value < best_value:
                best_value = value
                best_action = action

            if value <= alpha:
                return value, action

            beta = min(beta, value)

        return best_value, best_action

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -99999999
        beta = 99999999
        _, best_action = self.max_value(gameState, 0, self.depth, alpha, beta)
        return best_action
