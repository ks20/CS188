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

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in xrange(0, self.iterations):
            stateAndNewVs = []
            for state in states:
                possibleActions = self.mdp.getPossibleActions(state)
                vals = []
                for action in possibleActions:
                    vals += [self.getQValue(state, action)]
                if vals:
                    newVal = max(vals)
                else:
                    newVal = self.values[state]
                stateAndNewVs += [[state, newVal]]
            for sAndV in stateAndNewVs:
                state = sAndV[0]
                newVal = sAndV[1]
                self.values[state] = newVal



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        Qval = 0
        for transition in transitionStatesAndProbs:
            newState = transition[0]
            prob = transition[1]
            if self.mdp.isTerminal(state):
                reward = 0
            else:
                reward = self.mdp.getReward(state, action, newState)
            Qval += prob*(reward + self.discount*self.getValue(newState))
        return Qval
        "*** YOUR CODE HERE ***"
        """util.raiseNotDefined()"""

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        possibleActions = self.mdp.getPossibleActions(state)
        Qvals = []
        for action in possibleActions:
            Qvals += [self.getQValue(state, action)]
        return possibleActions[Qvals.index(max(Qvals))]
        "*** YOUR CODE HERE ***"
        """util.raiseNotDefined()"""

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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)
        for i in xrange(0, self.iterations):
            currState = states[i % numStates]
            if self.mdp.isTerminal(currState):
                continue
            possibleActions = self.mdp.getPossibleActions(currState)
            val = None
            for action in possibleActions:
                val = max(val, self.getQValue(currState, action))
            self.values[currState] = val


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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = {}
        for state in states:
            possibleActions = self.mdp.getPossibleActions(state)
            for action in possibleActions:
                transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                for transition in transitionStatesAndProbs:
                    if transition[1] > 0.0:
                        if transition[0] in predecessors:
                            predecessors[transition[0]].add(state)
                        else:
                            newSet = set()
                            newSet.add(state)
                            predecessors[transition[0]] = newSet

        pqueue = util.PriorityQueue()
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            possibleActions = self.mdp.getPossibleActions(state)
            maxQval = None
            for action in possibleActions:
                maxQval = max(maxQval, self.getQValue(state, action))
            currVal = self.getValue(state)
            diff = abs(currVal-maxQval)
            pqueue.update(state, -diff)

        for i in xrange(0, self.iterations):
            if pqueue.isEmpty():
                break
            state = pqueue.pop()
            if not self.mdp.isTerminal(state):
                possibleActions = self.mdp.getPossibleActions(state)
                newVal = None
                for action in possibleActions:
                    newVal = max(newVal, self.getQValue(state, action))
                self.values[state] = newVal
                if not predecessors[state]:
                    continue
                for p in predecessors[state]:
                    possibleActions = self.mdp.getPossibleActions(p)
                    maxQval = None
                    for action in possibleActions:
                        maxQval = max(maxQval, self.getQValue(p, action))
                    diff = abs(maxQval - self.values[p])
                    if diff > self.theta:
                        pqueue.update(p, -diff)

