# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random as rd
#%%

class MDP():
	""" 
		Defines an Markov Decision Process containing:
	
		- States, s
		- Actions, a
		- Rewards, r(s,a)
		- Transition Matrix, t(s,a,_s)

		Includes a set of abstract methods for extended class will
		need to implement.
	
	"""

	def __init__(self, NUM_STATES, discount=.99, epsilon=0.0001):
		
		self.NUM_STATES = NUM_STATES
		
		self.Graph = None

		self.states = None
		self.actions = None
		self.rewards = None
		self.transitions = None

		self.discount = discount
		self.epsilon = epsilon

		self.values = None
		self.policy = None

		self.createModel()
		self.terminalStates = [self.NUM_STATES - 1]

	def createModel(self):
		"""
			Create Graph representation and MDP variables with number of states designated
		"""
		self.Graph = nx.erdos_renyi_graph(self.NUM_STATES, 0.5, directed=False)
		
		# Random weights
		for u, v in self.Graph.edges():
			self.Graph.edges[u, v]['weight'] = rd.randint(1, 11)

		# Initializing MDP variables

		self.states = np.array(self.Graph.nodes) # S = State list
		self.actions = lambda s: list(self.Graph[s].keys()) # A(s) = Action function
		self.rewards = np.zeros((self.NUM_STATES, self.NUM_STATES)) # R = Reward list

		for s in self.states:
			for a in self.actions(s):
				self.rewards[s, a] = self.Graph[s][a]['weight']

		self.transitions = np.zeros((self.NUM_STATES, self.NUM_STATES, self.NUM_STATES))

		for s in self.states:
			for s_ in self.states:
				for a in self.actions(s):
					if s_ == a:
						self.transitions[s_, s, a] = 1

	def isTerminal(self, state):
		"""
			Checks if MDP is in terminal state.
		"""
		return state in self.terminalStates
	
	def addTerminalStates(self, states):
		"""
			Add new states how terminal state
		"""
		self.terminalStates += list(states)

	def getReward(self, state_, state, action, stochastic=False):
		"""
			Gets reward for transation from state->action->nextState
			choice if this reward is deterministic or stochastic
		"""
		return np.random.exponential(4) if stochastic else self.rewards[state, action]

	def valueIteration(self):
		"""
			Performs value iteration to populate the values of all states in
			the MDP. 
	
		"""

		self.values = np.zeros(self.NUM_STATES)
		self.qualities = np.zeros((self.NUM_STATES, self.NUM_STATES))
		self.policy = np.zeros(self.NUM_STATES)

		convergencia = []
		count = 0

		while True and count < 200:
			oldValues = np.copy(self.values)
			for s in self.states:
				if self.isTerminal(s):
					continue
				for a in self.actions(s):
					self.qualities[s, a] = self.getReward(a, s, a, True) + self.discount * \
						np.dot(self.transitions[:, s, a], self.values)
				
				self.values[s] = np.min([value for value in self.qualities[s, :] if value != 0])
			
			convergencia.append(np.max(np.abs(self.values - oldValues)))
			count += 1
			if np.max(np.abs(self.values - oldValues)) <= self.epsilon:
				break
			
			plt.plot(convergencia)
			plt.show

		
		self.extractPolicy()

	def extractPolicy(self):
		for s in self.states:
			argmin = np.argmax(self.qualities[s, :])
			for i in range(self.NUM_STATES):
				if self.qualities[s, i] == 0:
					continue
				elif self.qualities[s, i] <= self.qualities[s, argmin]:
					argmin = i
			self.policy[s] = argmin
	
	def policyTest(self, initialState=0):
		currentState = initialState
		history = [currentState]

		while not self.isTerminal(currentState):
			currentState = int(self.policy[currentState])
			history.append(currentState)
		
		path = ''
		cost = 0
		start = initialState
		for destination in history[1:]:
			path += f' => {destination}'
			cost += self.rewards[start, destination]
			start = destination

		print(f"Custo = {cost}\nHistórico: {initialState}{path}" )
		print(f'Dijkstra = {nx.dijkstra_path(self.Graph, initialState, start)}')

	def drawGraph(self):
		pos = nx.spring_layout(self.Graph)
		nx.draw(self.Graph, pos, with_labels=True)
		edge_labels = nx.get_edge_attributes(self.Graph, 'weight')
		nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels)

# %%
if __name__ == "__main__":
	mdp = MDP(7)
	# mdp.drawGraph()
	mdp.valueIteration()
	
	print('Values: ', mdp.values)
	print('Policy: ',  mdp.policy)
	# mdp.policyTest()
	


# %%
