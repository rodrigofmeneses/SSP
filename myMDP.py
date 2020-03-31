# Joey Velez-Ginorio
# MDP Implementation
# ---------------------------------
# - Includes BettingGame example


import numpy as np
import networkx as nx
import random


class MDP(object):
	""" 
		Defines an Markov Decision Process containing:
	
		- States, s 
		- Actions, a
		- Rewards, r(s,a)
		- Transition Matrix, t(s,a,_s)

		Includes a set of abstract methods for extended class will
		need to implement.

	"""

	
	
	def __init__(self, states=None, actions=None, rewards=None, transitions=None, 
				discount=.9, tau=.01, epsilon=.01):

		self.s = np.array(states)
		self.a = np.array(actions)
		self.r = np.array(rewards)
		self.t = np.array(transitions)
		
		self.discount = discount
		self.tau = tau
		self.epsilon = epsilon

		# Value iteration will update this
		self.values = None

	
	def isTerminal(self, state):
		"""
			Checks if MDP is in terminal state.
		"""
		raise NotImplementedError()

	def getTransitionStatesAndProbs(self, state, action):
		"""
			Returns the list of transition probabilities
		"""
		return self.t[state][action][:]

	def getReward(self, state, action):
		"""
			Gets reward for transition from state->action->nextState.
		"""
		return self.r[state][action]


	def takeAction(self, state, action):
		"""
			Take an action in an MDP, return the next state

			Chooses according to probability distribution of state transitions,
			contingent on actions.
		"""
		return np.random.choice(self.s, p=self.getTransitionStatesAndProbs(state, action))	


class StochasticShortestPath(MDP):
    
	"""
		Definição do Caminho Mínimo Estocástico:

		Problema: Dado um grafo G, onde não se sabe previamente quais os custos
		de suas arestas. Há um agente responsável por cumprir o objetivo de,
		a partir de um nó inicial alcançar o nó destino, e ele faz isso ao 
		escolher dentre as possíveis distribuições de probabilidade para encontrar
		qual minimiza o custo desse caminho.

		Parametros:
			states: nós do grafo
			actions: escolher a distribuição de probabilidade em cada vértice
			rewards: custos da aresta 
			transitions: 

    """

	def __init__(self):
		
		MDP.__init__(self)
		self.setStochasticShortestPath()
		self.valueIteration()


	def setStochasticShortestPath(self):
		# New Graph
		self.G = nx.Graph()
		
		# Add nodes
		self.G.add_nodes_from(range(6))

		# Add edges
		self.G.add_edges_from([(0, 1, {'w': 1}), (0, 2, {'w': 1}), (1, 2, {'w': 2}), \
		(1, 3, {'w': 1}), (2, 4, {'w': 2}), (3, 4, {'w': 1}), (3, 5, {'w': 3}), \
		(4, 5, {'w': 1})])

		# Initialize all possible states
		self.s = np.array(self.G.nodes)

		# Initialize all possible actions
		self.a = np.array(self.G.nodes)

		# Initialize rewards
		self.r = np.zeros((self.G.number_of_edges(), self.G.number_of_edges()))
		for i, j in self.G.edges:
			self.r[i, j] = self.G[i][j]['w']

		# Incidence Matrix
		self.r = self.r + self.r.T

		# Initialize transitions		
		self.t = np.zeros((len(self.s), len(self.a), len(self.s)))
		
		# All transitions probabilities are 1
		for state, action in self.G.adjacency():
			for elem in action:
				self.t[state, elem, elem] = 1.0
	

	def valueIteration(self):
		"""
			Performs value iteration to populate the values of all states in
			the MDP. 

		"""

		# Initialize V_0 to zero
		self.values = np.zeros(len(self.s))
		self.policy = np.zeros([len(self.s), len(self.a)])

		# Loop until convergence
		while True:
			# To be used for convergence check
			oldValues = np.copy(self.values)

			# For all States
			for s in self.G.nodes:
				action_values = list()
				# All actions
				for a in self.G.adj[s].keys():
					# Update Bellman's Equation
					action_values.append(self.r[s, a] + self.discount * \
						np.dot(self.t[s, a, :], self.values))
				self.values[s] = np.min(action_values)					
				
			# Check Convergence
			if np.max(np.abs(self.values - oldValues)) <= self.epsilon:
				break



ssp = StochasticShortestPath()
print('values', ssp.values)
print('policy', ssp.policy)
