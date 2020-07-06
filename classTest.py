# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random as rd
import os
import time
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

	def __init__(self, numNodes=10, density=1, discount=.99, epsilon=0.000001):
		
		self.numNodes = numNodes
		self.density = density
		
		self.Graph = None

		self.states = None
		self.actions = None
		self.rewards = None
		self.transitions = None

		self.values = None
		self.policy = None

		self.discount = discount
		self.epsilon = epsilon

		self.terminal_states = []
		self._createModel()

	'''
		Private Methods
	'''

	def _createModel(self):
		"""
			Create Graph representation and MDP variables with number of states designated
		"""
		# Graph with density probability of create a edge
		self.Graph = nx.erdos_renyi_graph(self.numNodes, self.density, directed=False)

		# Guarantee of connectivy
		for node in range(self.numNodes):
			if not self.Graph[node]:
				except_node_list = [x for x in range(self.numNodes) if x != node]
				self.Graph.add_edge(node, np.random.choice(except_node_list))
		
		# Random weights
		for u, v in self.Graph.edges():
			self.Graph.edges[u, v]['weight'] = rd.randint(1, 11)

		# Initializing MDP variables

		self.states = np.array(self.Graph.nodes) # S = State list
		self.actions = lambda s: list(self.Graph[s].keys()) # A(s) = Action function
		self.rewards = np.zeros((self.numNodes, self.numNodes)) # R = Reward list

		for s in self.states:
			for a in self.actions(s):
				self.rewards[s, a] = self.Graph[s][a]['weight']

		self.transitions = np.zeros((self.numNodes, self.numNodes, self.numNodes))

		for s in self.states:
			for s_ in self.states:
				for a in self.actions(s):
					if s_ == a:
						self.transitions[s_, s, a] = 1
		
		self.terminal_states = []
		self._addTerminalStates(np.random.choice(list(self.states)[1:]))

	def _is_terminal(self, state):
		"""
			Checks if MDP is in terminal state.
		"""
		return state in self.terminal_states
	
	def _addTerminalStates(self, state):
		"""
			Add new state how terminal state
		"""
		self.terminal_states.append(state)
		

	def _getReward(self, state_, state, action, stochastic=False):
		"""
			Gets reward for transation from state->action->nextState
			choice if this reward is deterministic or stochastic
		"""
		return np.random.exponential(scale=0.5) * self.rewards[state, action]\
            if stochastic else self.rewards[state, action]

	def _extract_policy(self):
		for s in self.states:
			argmin = np.argmax(self.qualities[s, :])
			for i in range(self.numNodes):
				if self.qualities[s, i] == 0:
					continue
				elif self.qualities[s, i] <= self.qualities[s, argmin]:
					argmin = i
			self.policy[s] = argmin
	
	def _policy_path(self, initial_state):
		'''
			Follow policy and return the history
		'''
		current_state = initial_state
		history = [current_state]
		while not self._is_terminal(current_state):
			current_state = int(self.policy[current_state])
			history.append(current_state)
		
		return history
	
	def _policy_path_length(self, initial_state):
		'''
			Follow policy and return total cost
		'''
		current_state = initial_state
		next_state = int(self.policy[current_state])
		cost = 0
		while True:
			cost += self.Graph[current_state][next_state]['weight']
			if self._is_terminal(next_state):
				return cost
			current_state, next_state = next_state, int(self.policy[next_state])

	'''
		Public Methods
	'''
	
	def value_iteration(self):
		"""
			Performs value iteration to populate the values of all states in
			the MDP. 
	
		"""

		self.values = np.zeros(self.numNodes)
		self.qualities = np.zeros((self.numNodes, self.numNodes))
		self.policy = np.zeros(self.numNodes)

		while True:
			oldValues = np.copy(self.values)
			for s in self.states:
				if self._is_terminal(s):
					continue
				for a in self.actions(s):
					self.qualities[s, a] = self._getReward(a, s, a, False) + self.discount * \
						np.dot(self.transitions[:, s, a], self.values)
				
				self.values[s] = np.min([value for value in self.qualities[s, :] if value != 0])
			
			if np.max(np.abs(self.values - oldValues)) <= self.epsilon:
				break
			
		
		self._extract_policy()

	def simulation(self, sample_nodes_density):

		for nodes, density in sample_nodes_density:
			path = f'Experimentos\\{nodes}_{density}\\'
			if not os.path.exists(path):
				os.mkdir(path)
			else:
				os.remove(path + 'resultados.txt')

			for sample in range(1, 11):
				self.numNodes = nodes
				self.density = density
				self._createModel()

				nx.write_adjlist(self.Graph, path + f'adjlist_{nodes}_{density}_{sample}.txt')
				nx.write_edgelist(self.Graph, path + f'edgelist_{nodes}_{density}_{sample}.txt')

				start_value_iteration = time.time()
				self.value_iteration()
				end_value_iteration = time.time()

				start_dijkstra = time.time()
				dijkstra_length = nx.dijkstra_path_length(self.Graph, 0, self.terminal_states[0])
				end_dijkstra = time.time()

				with open(path + 'resultados.txt', 'a') as result_file:
					result = f'{sample} '
					result += str(self._policy_path_length(0) == dijkstra_length) + ' '
					result += str(end_value_iteration - start_value_iteration) + ' '
					result += str(end_dijkstra - start_dijkstra) + '\n'
					result_file.write(result)

	def drawGraph(self):
		pos = nx.spring_layout(self.Graph)
		nx.draw(self.Graph, pos, with_labels=True)
		edge_labels = nx.get_edge_attributes(self.Graph, 'weight')
		nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels)

# %%
if __name__ == "__main__":
	mdp = MDP(10)
	mdp.simulation([(10, 0.7), (30, 0.6), (50, 0.5)])
	# mdp.value_iteration()
	# print('Values: ', mdp.values)
	# print('Policy: ',  mdp.policy)
	# mdp._policy_path()

# %%
