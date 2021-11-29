# Dvir Asaf 313531113
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Get a list of all the positions where the food is located on board
        food_pos = successorGameState.getFood().asList()

        # Init the minimum distance between the Pacman's position and the food to "infinity"
        distance_to_nearest_food = float("inf")

        # Going over each food on the board and search for the the one with the shortest distance to the pacman
        for food in food_pos:
            distance_to_food = manhattanDistance(newPos, food)
            distance_to_nearest_food = min(distance_to_nearest_food, distance_to_food)

        # Check if there are any ghosts around the pacman
        for ghost in successorGameState.getGhostPositions():
            # The ghost is too close, we should return minimum score, therefore we'll return -infinity
            dist_to_ghost = manhattanDistance(newPos, ghost)
            if (2 > dist_to_ghost):
                return -float('inf')

        # The idea behind the score calculation: The smaller the distance between the pacman and the food, the higher
        # the score, since dividing 1 by that number.
        return successorGameState.getScore() + 1.0 / distance_to_nearest_food

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
        "*** YOUR CODE HERE ***"
        # agent index set to 0 - pacman
        # depth is zero at beginning
        move = self.MaxValue(gameState, 0, 0)
        # return minmax action
        return move[0]

    def MaxValue(self, gameState, agentIndex, depth):
        '''
        The pacman needs to get the highest possible score
        '''
        best_pacman_move = ("max", -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            # The possible actions according to the agent's index
            successor_move, successor_score = self.get_successor_move(action, agentIndex, depth, gameState)

            # update the highest possible score
            if successor_score > best_pacman_move[1]:
                best_pacman_move = successor_move
        return best_pacman_move

    def get_successor_move(self, action, agentIndex, depth, gameState):
        game_state_successor = gameState.generateSuccessor(agentIndex, action)
        depth_successor = depth + 1
        agent_index_successor = depth_successor % gameState.getNumAgents()
        successor_score = self.MinMaxAlg(game_state_successor, agent_index_successor, depth_successor)
        successor_move = (action, successor_score)
        return successor_move, successor_score

    def MinMaxAlg(self, gameState, agentIndex, depth):
        # If in the current depth is final we should evaluate the state in which it was found.
        if gameState.isLose() or \
                self.depth * gameState.getNumAgents() == depth or \
                gameState.isWin():
            return self.evaluationFunction(gameState)

        # Pacman: will try to get the maximum score for himself
        if agentIndex == 0:
            return self.MaxValue(gameState, agentIndex, depth)[1]

        # Ghosts: they will try to get the minimum score for the pacman.
        else:
            return self.MinValue(gameState, agentIndex, depth)[1]

    def MinValue(self, gameState, agentIndex, depth):
        """
        Ghosts want to make the pacman get the lowest score
        """

        best_ghosts_move = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            successor_move, successor_score = self.get_successor_move(action, agentIndex, depth, gameState)
            # update the lowest possible score
            if successor_score < best_ghosts_move[1]:
                best_ghosts_move = successor_move
        return best_ghosts_move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float("inf")
        # agent index set to 0 - pacman
        # depth is zero at beginning
        move = self.MaxValue(gameState, 0, 0, -inf, inf)
        # return minmax action
        return move[0]

    def get_successor_move(self, action, agentIndex, alpha, beta, depth, gameState):
        game_state_successor = gameState.generateSuccessor(agentIndex, action)
        depth_successor = depth + 1
        agent_index_successor = depth_successor % gameState.getNumAgents()
        successor_score = self.AlphaBetaAlg(game_state_successor, agent_index_successor, depth_successor, alpha, beta)
        successor_move = (action, successor_score)
        return successor_move, successor_score

    def AlphaBetaAlg(self, gameState, agentIndex, depth, alpha, beta):
        # If the current depth is final we should evaluate the state in which it was found.
        if gameState.isLose() or \
                self.depth * gameState.getNumAgents() == depth or \
                gameState.isWin():
            return self.evaluationFunction(gameState)
        # Pacman: will try to maximize alpha
        if agentIndex == 0:
            return self.MaxValue(gameState, agentIndex, depth, alpha, beta)[1]
        # Ghosts: they will try to minimize (beta)
        else:
            return self.MinValue(gameState, agentIndex, depth, alpha, beta)[1]

    def MaxValue(self, gameState, agentIndex, depth, alpha, beta):
        best_pacman_move = ("max", -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            successor_move, successor_score = self.get_successor_move(action, agentIndex, alpha, beta, depth, gameState)

            # update the highest possible score
            if successor_score > best_pacman_move[1]:
                best_pacman_move = successor_move

            # Max pruning
            if beta < best_pacman_move[1]:
                return best_pacman_move
            else:
                alpha = max(best_pacman_move[1], alpha)

        return best_pacman_move

    def MinValue(self, gameState, agentIndex, depth, alpha, beta):
        best_ghosts_move = ("min", float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            successor_move, successor_score = self.get_successor_move(action, agentIndex, alpha, beta, depth, gameState)

            # update the lowest possible score
            if successor_score < best_ghosts_move[1]:
                best_ghosts_move = successor_move

            # Min pruning
            if alpha > best_ghosts_move[1]:
                return best_ghosts_move
            else:
                beta = min(best_ghosts_move[1], beta)

        return best_ghosts_move
