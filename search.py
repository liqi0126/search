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

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
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


def depth_first_exploration(problem, pos, explored_set, actions):
    for newstate in problem.expand(pos):
        child, action, _ = newstate
        if child in explored_set:
            continue

        actions.push(action)
        explored_set.append(child)
        if problem.isGoalState(child):
            return True
        if depth_first_exploration(problem, child, explored_set, actions):
            return True
        else:
            actions.pop()

    return False

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    explored_set = []
    actions = util.Stack()
    start = problem.getStartState()
    explored_set.append(start)

    if depth_first_exploration(problem, start, explored_set, actions):
        return actions.list
    else:
        raise RuntimeError


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    explored_set = []
    frontier = util.Queue()
    parents = {}

    start = problem.getStartState()
    explored_set.append(start)
    frontier.push(start)
    parents[start] = None

    while not frontier.isEmpty():
        pos = frontier.pop()
        if problem.isGoalState(pos):
            actions = []
            while parents[pos] is not None:
                parent_node, action = parents[pos]
                actions.append(action)
                pos = parent_node
            actions.reverse()
            return actions

        for newstate in problem.expand(pos):
            child, action, _ = newstate
            if child in explored_set:
                continue
            explored_set.append(child)
            frontier.push(child)
            parents[child] = (pos, action)

    raise RuntimeError


def nullHeuristic(state, problem=None):
    """
    A example of heuristic function which estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial. You don't need to edit this function
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    g_costs = {}
    frontier = util.PriorityQueue()
    parents = {}

    start = problem.getStartState()
    g_costs[start] = 0
    frontier.push(start, heuristic(start, problem))
    parents[start] = None

    while not frontier.isEmpty():
        pos = frontier.pop()
        if problem.isGoalState(pos):
            actions = []
            while parents[pos] is not None:
                parent_node, action = parents[pos]
                actions.append(action)
                pos = parent_node
            actions.reverse()
            return actions

        for newstate in problem.expand(pos):
            child, action, _ = newstate
            g_cost = g_costs[pos] + problem.getActionCost(pos, action, child)
            if child not in g_costs or g_costs[child] > g_cost:
                g_costs[child] = g_cost
                parents[child] = (pos, action)
                frontier.update(child, g_costs[child] + heuristic(child, problem))

    raise RuntimeError

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch