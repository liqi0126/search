from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


# def scoreEvaluationFunction(currentGameState):
#     """
#     This default evaluation function just returns the score of the state.
#     The score is the same one displayed in the Pacman GUI.
#     """
#     return currentGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsule = currentGameState.getCapsules()
    # highest score for a winning state
    if currentGameState.isWin():
        return 9999999999

    # worst case if pacman and ghost position are the same
    # but ghost is not scared
    for state in currentGhostStates:
        if state.getPosition() == currentPos and state.scaredTimer == 1:
            return -99999

    score = 0

    # chase food - food gobbling
    # better score for state with food near and ghosts far
    # check distance of food from the Pacman
    foodDistance = [util.manhattanDistance(currentPos, food) \
                    for food in currentFood]
    nearestFood = min(foodDistance)
    # nearer food should have more weightage - take inverse
    score += float(1 / nearestFood)
    # subtract the no of food left and proportional weight to this as we want to Pick
    # state with less food leftover
    score -= len(currentFood)

    # chase capsule - pellet nabbing
    # score for capsules
    if currentCapsule:
        capsuleDistance = [util.manhattanDistance(currentPos, capsule) \
                           for capsule in currentCapsule]
        nearestCapsule = min(capsuleDistance)
        # near capsule better
        score += float(10 / nearestCapsule)

    # chase ghost when ghost is scared else avoid - ghost hunting
    currentGhostDistances = [util.manhattanDistance(currentPos, ghost.getPosition()) \
                             for ghost in currentGameState.getGhostStates()]
    nearestCurrentGhost = min(currentGhostDistances)
    scaredTime = sum(currentScaredTimes)
    # farther ghosts are better
    if nearestCurrentGhost >= 1:
        if scaredTime < 0:
            score -= 1 / nearestCurrentGhost
        else:
            score += 1 / nearestCurrentGhost

    return currentGameState.getScore() + score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
                value, _ = self.min_value(next_state, index + 1, depth)
            else:
                value, _ = self.max_value(next_state, index + 1, depth)

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
                value, _ = self.max_value(next_state, 0, depth - 1)
            else:
                value, _ = self.min_value(next_state, index + 1, depth)

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
                value, _ = self.min_value(next_state, index + 1, depth, alpha, beta)
            else:
                value, _ = self.max_value(next_state, index + 1, depth, alpha, beta)

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
                value, _ = self.max_value(next_state, 0, depth - 1, alpha, beta)
            else:
                value, _ = self.min_value(next_state, index + 1, depth, alpha, beta)

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
