import numpy as np
import matplotlib.pyplot as plt

# grid
grid = [
    ['O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'X', 'X', 'O'],
    ['X', 'X', 'O', 'O', 'O'],
    ['O', 'X', 'O', 'X', 'X'],
    ['O', 'O', 'O', 'X', 'X'],
    ['X', 'X', 'O', 'O', 'X'],
    ['G', 'O', 'O', 'O', 'X']
]

# rewards
rewards = {
    'O': 0,  # free space
    'X': float('-inf'),  # obstacle
    'G': 10  # goal
}

# actions
actions = [
    (-1, 0),  # up
    (1, 0),  # down
    (0, -1),  # left
    (0, 1)  # right
]

# transition probabilities
transition_probs = {
    (-1, 0): [(0.8, 0), (0.2, 1)],    # up: [(probability, index of next state)]
    (1, 0): [(0.8, 1), (0.2, 0)],      # down: [(probability, index of next state)]
    (0, -1): [(0.8, 2), (0.2, 3)],     # left: [(probability, index of next state)]
    (0, 1): [(0.8, 3), (0.2, 2)]       # right: [(probability, index of next state)]
}

# initialize value function => V
rows = len(grid)
cols = len(grid[0])
V = np.zeros((rows, cols))

convergence_threshold = 1e-6  # convergence threshold
discount_factor = 0.9  # discount factor => gamma
converged = False

# value-iterative
num_of_iterations = 0
while not converged:
    V_new = np.copy(V)
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'G':
                # G is terminal state
                V_new[i][j] = 10
            if grid[i][j] != 'X':  # exclude obstacles
                max_value = float('-inf')
                for action in actions:
                    next_i = i + action[0]
                    next_j = j + action[1]
                    if 0 <= next_i < rows and 0 <= next_j < cols and grid[next_i][next_j] != 'X':
                        values = []
                        for prob, next_state_idx in transition_probs[action]:
                            value = prob * (rewards[grid[next_i][next_j]] + discount_factor * V[next_i][next_j])
                            values.append(value)
                        action_value = sum(values)
                        max_value = max(max_value, action_value)
                V_new[i][j] = max_value
    if np.max(np.abs(V_new - V)) < convergence_threshold:
        converged = True
    V = np.copy(V_new)
    num_of_iterations += 1

# G is terminal state
V[rows-1][0] = 10

# determine optimal policy
policy = np.empty((rows, cols), dtype=str)
for i in range(rows):
    for j in range(cols):
        if grid[i][j] != 'X' and grid[i][j] != 'G':
            max_value = float('-inf')
            best_action = None
            for k, action in enumerate(actions):
                next_i = i + action[0]
                next_j = j + action[1]
                if 0 <= next_i < rows and 0 <= next_j < cols and grid[next_i][next_j] != 'X':
                    values = []
                    for prob, next_state_idx in transition_probs[action]:
                        value = prob * (rewards[grid[next_i][next_j]] + discount_factor * V[next_i][next_j])
                        values.append(value)
                    action_value = sum(values)
                    if action_value > max_value:
                        max_value = action_value
                        best_action = k
            if best_action is not None:
                policy[i][j] = ['Up', 'Down', 'Left', 'Right'][best_action]
            else:
                policy[i][j] = '-'
        else:
            policy[i][j] = '-'

# print optimal policy (for question1)
print('optimal policy: ')
for row in policy:
    print(' '.join(row))

print('\n**************************************************************\n')


# visualize value matrix V (for question2)
plt.imshow(V, cmap='coolwarm', interpolation='nearest')

# add value labels to each state
for i in range(rows):
    for j in range(cols):
        plt.text(j, i, f'{V[i][j]:.3f}', color='black', ha='center', va='center')

plt.colorbar(label='Value')
plt.title('state-values')
plt.show()

print('State-Value Function (V):')
np.set_printoptions(suppress=True, precision=30, formatter={'float': '{:0.2e}'.format})
print(V)

print('\n**************************************************************\n')

print('num_of_iterations: ', num_of_iterations)