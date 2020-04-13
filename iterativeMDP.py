# %%
import numpy as np
import networkx as nx
import random as rd
#%%
# Important data for MDP

s = None # all states
a = None # all actions
r = None # all rewards
t = None # all transitions probabilities

discount = .95 # discount factor, for future rewards
epsilon = .0001 # epsilon, for convergence

values = None # value function
policy = None # policy learned
q = None # for q(s, a) table

# %%
# Stochastic shortest path problem

def createGraph(n, m, s, a, r, t):
    # n = number of nodes
    # m = number of edges

    # Creating graphs
    # Random graph with n nodes, m edges
    G = nx.gnm_random_graph(n, m)
    # Complete graph with n nodes
    # G = nx.complete_graph(n)

    # Fill all edges with weights in range (0, 11)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = rd.randint(1, 11)

    # Initializzing MDP variables
    s = np.array(G.nodes) # state list
    a = np.array(G.nodes) # action list
    r = np.zeros((len(s), len(a), len(s))) # state s, action a, reach state s_

    # fill reward table witch edge weights
    # r(s, a, s_) = current state s, take action a, reach state s_, reward
    for u in s:
        for v in G[u].keys():	
            for action in G[u].keys():
                r[u, v, action] = G[u][action]['weight']

    # Transitions probabilities
    # t(s, a, s_) = current state s, take action a, reach state s_, probability
    t = np.zeros((len(s), len(a), len(s)))

    # All transitions probabilities are 1, deterministic
    for state, action in G.adjacency():
        for elem in action:
            t[state, elem, elem] = 1.0
    
    return G, s, a, r, t

# %%
def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    # Labels of edges
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # Decorated graph with labels of edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
# %%

# Value iteration

def valueIteration(s, a, r, t):
	# Initialize V[s] e Q[s, a] to zero    
    values = np.zeros(len(s))
    q = np.zeros((len(s), len(a)))
    policy = np.zeros(len(s))

    end_state = s[-1]

    # while not converge
    while True:
        oldValues = np.copy(values)
        # for all states
        for state in s:
            # for all avaibles actions
            if state == end_state:
                values[state] = 0
                break
            for action in list(G[state].keys()):
                # update with bellman equation
                q[state, action] = r[state, action, action] + discount * \
                np.dot(t[state, action, :], values)
            # shortest path, we want less reward as possible
            # 
            values[state] = np.min([x for x in q[state, :] if x != 0])
        
        # Check Convergence
        if np.max(np.abs(values - oldValues)) <= epsilon:
            break
        
    for state in s:
        if state == end_state:
            break
        argmin = np.argmax(q[state, :])
        for i in range(len(s)):
            if q[state, i] == 0:
                continue
            elif q[state, i] <= q[state, argmin]:
                argmin = i
        policy[state] = argmin
        
    return values, q, policy


#%%
n = 6
m = 10
G, s, a, r, t = createGraph(n, m, s, a, r, t)
drawGraph(G)
# %%
values, q, policy = valueIteration(s, a, r, t)

print('values', values)
print('q', q)
print('policy', policy)
print('shoertest path', nx.shortest_path(G, 0, n - 1, 'weight'))

# %%
