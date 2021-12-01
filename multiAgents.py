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


from game import Directions
import random, util
import numpy as np

from game import Agent

def getLegalActionsNoStop(index, gameState):
    possibleActions = gameState.getLegalActions(index)
    if Directions.STOP in possibleActions:
        possibleActions.remove(Directions.STOP)
    return possibleActions


class ReflexAgent(Agent):

    def getAction(self, gameState):

        legalActions = gameState.getLegalActions()
        scores = [evaluationFunction(gameState, action) for action in legalActions]
        maxBestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == maxBestScore]
        chosenIndex = random.choice(bestIndices)

        return legalActions[chosenIndex]


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):

    def minimax(self, agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
              getLegalActionsNoStop(0, gameState))
        else:
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
              getLegalActionsNoStop(agent, gameState))

    def getAction(self, gameState):

        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action
          in possibleActions]
        maxAction = max(action_scores)
        maxIndices = [index for index in range(len(action_scores)) if action_scores[index] == maxAction]
        chosenIndex = random.choice(maxIndices)
        return possibleActions[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):

    def alphabeta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            value = -999999
            for action in getLegalActionsNoStop(agent, gameState):
                value = max(value, self.alphabeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            nextAgent = agent + 1 
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            for action in getLegalActionsNoStop(agent, gameState):
                value = 999999
                value = min(value, self.alphabeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def getAction(self, gameState):

        possibleActions = getLegalActionsNoStop(0, gameState)
        alpha = -999999
        beta = 999999
        action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), alpha, beta) for action
          in possibleActions]
        maxAction = max(action_scores)
        maxIndices = [index for index in range(len(action_scores)) if action_scores[index] == maxAction]
        chosenIndex = random.choice(maxIndices)
        return possibleActions[chosenIndex]


class ExpectimaxAgent(MultiAgentSearchAgent):

    def expectimax(self, agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            return max(self.expectimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
              getLegalActionsNoStop(0, gameState))
        else:
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            return sum(self.expectimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
              getLegalActionsNoStop(agent, gameState)) / float(len(getLegalActionsNoStop(agent, gameState)))

    def getAction(self, gameState):

        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.expectimax(0, 0, gameState.generateSuccessor(0, action)) for action
          in possibleActions]
        maxAction = max(action_scores)
        maxIndices = [index for index in range(len(action_scores)) if action_scores[index] == maxAction]
        chosenIndex = random.choice(maxIndices)
        return possibleActions[chosenIndex]


def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()


def evaluationFunction(currentGameState, action):

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newFoodList = np.array(newFood.asList())
    distanceToFood = [util.manhattanDistance(newPos, food) for food in newFoodList]
    minFoodDistance = 0
    if len(newFoodList) > 0:
        minFoodDistance = distanceToFood[np.argmin(distanceToFood)]

    ghostPositions = np.array(successorGameState.getGhostPositions())
    distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
    minGhostDistance = 0
    nearestGhostScaredTime = 0
    if len(ghostPositions) > 0:
        minGhostDistance = distanceToGhost[np.argmin(distanceToGhost)]
        nearestGhostScaredTime = newScaredTimes[np.argmin(distanceToGhost)]
        if minGhostDistance <= 1 and nearestGhostScaredTime == 0:
            return -999999
        if minGhostDistance <= 1 and nearestGhostScaredTime > 0:
            return 999999

    value = successorGameState.getScore() - minFoodDistance
    if nearestGhostScaredTime > 0:
        value -= minGhostDistance
    else:
        value += minGhostDistance
    return value

