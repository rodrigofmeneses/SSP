# %%
import numpy as np
import networkx as nx
import random as rd

#%%
# Important data for MDP

states = None # all states
actions = None # all actions(state)
rewards = None # all rewards
transitions = None # all transitions probabilities

discount = .95 # discount factor, for future rewards
epsilon = .0001 # epsilon, for convergence

values = None # value function
qualities = None # for q(s, a) table
policy = None # policy learned


# %%

# A função de valor é em cima de uma distribuição de probabilidade?
# decisão e ações
# decisão ... escolher uma das distribuições de prob disponíveis no nó
# ação ... escolher qual ação tomar...

def valueIteration(states, actions, rewards, transitions):

    # 1 - Initialization with variables with 0
    values = np.zeros(NUM_STATES) # v(s) function
    qualities = np.zeros((NUM_STATES, NUM_STATES)) # q(s, a) function
    policy = np.zeros(NUM_STATES) # pi(s) policy

    endStates = [states[-1]]

    # 2 - Iteration in bellman equation for search optmal value function
    while True:
        oldValues = np.copy(values) # for convergence

        for s in states:
            if s in endStates:
                continue
            for a in actions(s):
                qualities[s, a] = rewards[s, a] + discount * \
                    np.dot(transitions[:, s, a], values)
            
            values[s] = np.min([value for value in qualities[s, :] if value != 0])

        if np.max(np.abs(values - oldValues)) <= epsilon: # convergence test
            break
    
    # 3 - Extract policy from value function.
    for s in states:
        if s in endStates:
            continue
        argmin = np.argmax(qualities[s, :])
        for i in range(NUM_STATES):
            if qualities[s, i] == 0:
                continue
            elif qualities[s, i] <= qualities[s, argmin]:
                argmin = i
        policy[s] = argmin        

    return values, qualities, policy

#%%

def policyIteration(state, actions, rewards, transitions):
        
    # 1 - Initialization with a random policy
    values = np.zeros(NUM_STATES) # v(s) function
    qualities = np.zeros((NUM_STATES, NUM_STATES)) # q(s, a) function
    policy = [np.random.choice(actions(s)) for s in states] # random policy

    endStates = [states[-1]]

    while True:
        # 2 - Policy Evaluation

        while True:
            oldValues = np.copy(values)
            
            for s in states:
                if s in endStates:
                    continue
                values[s] = rewards[s, policy[s]] + discount * \
                    np.dot(transitions[:, s, policy[s]], values)

            if np.max(np.abs(values - oldValues)) <= epsilon: 
                break
                
        # 3 - Policy Improvement

        policyStable = True

        for s in states:
            oldAction = int(policy[s])
            for a in actions(s):
                qualities[s, a] = rewards[s, a] + discount * \
                    np.dot(transitions[:, s, a], values)
                
            policy[s] = np.argmin([x if x > 0 else 1000000 for x in qualities[s, :]])

            if oldAction != policy[s]:
                policyStable = False
        
        if policyStable:
            return values, qualities, policy

# %%

def createModel(n):
    # n = number of nodes

    # Creating graphs
    # Random graph with n nodes, m edges
    # G = nx.gnm_random_graph(n, m)
    # Create a edge with x of probability
    G = nx.erdos_renyi_graph(n, 0.5, directed=False)
    
    # Random weights
    for u, v in G.edges():
        G.edges[u, v]['weight'] = rd.randint(1, 11)

    # Initializing MDP variables

    states = np.array(G.nodes) # S = State list
    actions = lambda s: list(G[s].keys()) # A(s) = Action function
    rewards = np.zeros((len(states), len(states))) # R = Reward list

    for s in states:
        for a in actions(s):
            rewards[s, a] = G[s][a]['weight']

    transitions = np.zeros((len(states), len(states), len(states)))

    for s in states:
        for s_ in states:
            for a in actions(s):
                if s_ == a:
                    transitions[s_, s, a] = 1

    return G, states, actions, rewards, transitions

# %%

def policyTest(policy, initialState, endStates):
    currentState = initialState
    history = [currentState]

    while currentState not in endStates:
        currentState = int(policy[currentState])
        history.append(currentState)
    
    path = ''
    for p in history:
        path += f' => {p}'


    print(f"Histórico{path}" )
    
    return history

# %%
 
def drawGraph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

# %%

NUM_STATES = 10

G, states, actions, rewards, transitions = createModel(NUM_STATES)
drawGraph(G)

# %%
values, qualities, policy = policyIteration(states, actions, rewards, transitions)
policyTest(policy, 0, [NUM_STATES - 1])

# %%
values, qualities, policy = valueIteration(states, actions, rewards, transitions)
policyTest(policy, 0, [NUM_STATES - 1])
# %%