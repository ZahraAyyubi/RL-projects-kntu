import numpy as np

# Environment: states and actions and rewards
states = ['T1', 'A', 'START', 'B', 'C', 'D', 'E', 'T2']
actions = ['left', 'right']
rewards = {
    'A': -1,
    'START': -1,
    'B': -1,
    'C': -1,
    'D': -1,
    'E': -1,
    'T1': 0,
    'T2': 0
}

# Transition probabilities
transition_probs = {
    'A': {'left': {'T1': 0.2, 'B': 0.8}, 'right': {'B': 1.0}},
    'START': {'left': {'A': 1.0}, 'right': {'B': 1.0}},
    'B': {'left': {'A': 1.0}, 'right': {'C': 1.0}},
    'C': {'left': {'B': 1.0}, 'right': {'D': 1.0}},
    'D': {'left': {'C': 1.0}, 'right': {'E': 1.0}},
    'E': {'left': {'D': 1.0}, 'right': {'T2': 1.0}},
    'T1': {'left': {}, 'right': {}},
    'T2': {'left': {}, 'right': {}}
}


# choose an action based on a soft policy
def choose_action_soft_policy(state, q_values, epsilon):
    action_probs = np.ones(len(actions)) * epsilon / len(actions)
    best_action = np.argmax(list(q_values[state].values()))
    action_probs[best_action] += 1 - epsilon
    action = np.random.choice(actions, p=action_probs)
    return action


# Monte Carlo algorithm
def monte_carlo(gamma, num_episodes, epsilon):
    q_values = {state: {action: 0 for action in actions} for state in states}
    returns = {state: {action: [] for action in actions} for state in states}
    policy = {state: np.random.choice(actions) if state not in ['T1', 'T2'] else 'None' for state in states}

    for episode in range(num_episodes):
        states_visited = []
        actions_taken = []
        rewards_received = []

        # generate an episode
        state = 'START'
        while state not in ['T1', 'T2']:
            action = choose_action_soft_policy(state, q_values, epsilon)
            next_state = np.random.choice(list(transition_probs[state][action].keys()),
                                          p=list(transition_probs[state][action].values()))
            reward = rewards[next_state]
            states_visited.append(state)
            actions_taken.append(action)
            rewards_received.append(reward)
            state = next_state

        # update Q-values and policy
        G = 0
        for t in reversed(range(len(states_visited))):
            state_t = states_visited[t]
            action_t = actions_taken[t]
            reward_t = rewards_received[t]
            G = gamma * G + reward_t
            if state_t not in ['T1', 'T2']:  # skip terminal states
                returns[state_t][action_t].append(G)
                q_values[state_t][action_t] = np.mean(returns[state_t][action_t])
                best_action = np.argmax(list(q_values[state_t].values()))
                policy[state_t] = actions[best_action]

    return policy


# gamma values to test
gamma_values = [1.0, 0.5, 0.1]

# get optimal policy for each gamma value
for i, gamma in enumerate(gamma_values):
    policy = monte_carlo(gamma, num_episodes=1000, epsilon=0.4)

    print("optimal policy with gamma: ", gamma)
    print(policy)
    print('**********************************************************************************************************************')
