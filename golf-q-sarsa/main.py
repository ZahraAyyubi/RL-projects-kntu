import numpy as np
import random


# define environment Golf problem
class GolfEnvironment:
    # initialization for golf environment
    # we have four types of golf clubs with different default distances and precision values
    def __init__(self):
        self.initial_distance = 100
        self.clubs = {
            'woods': {'defaultDistance': 18, 'mean': 1, 'var': 0.25},
            'irons': {'defaultDistance': 12, 'mean': 1, 'var': 0.15},
            'hybrids': {'defaultDistance': 6, 'mean': 1, 'var': 0.05},
            'putter': {'defaultDistance': 3, 'mean': 1, 'var': 0}
        }
        # reset environment to initial state
        self.reset()

    # this function defines the transition dynamics of golf environment
    def step(self, action):
        club, power = action
        club_stats = self.clubs[club]

        # precision is a random variable dependent on the chosen golf club
        precision = np.random.normal(club_stats['mean'], club_stats['var'])

        # wind disturbance is a random variable that affects the golf shot
        wind_disturbance = np.random.normal(0, 3)

        # formula for calculating the distance the ball will travel
        distance = (power * club_stats['defaultDistance'] * precision) + round(
            (1 - precision) * wind_disturbance)

        # current distance is updated by subtracting the distance the ball has travelled from the current distance
        # round => discrete values of distance, abs => if the ball passes through the hole
        self.current_distance = min(100, abs(round(self.current_distance - distance)))

        # check if the ball has reached the hole, if yes, then game is done and a reward of 100 is given. Otherwise, a penalty of -1 is applied for each step
        if self.current_distance == 0:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        # return new state, reward and the 'done' flag
        return self.current_distance, reward, done

    # this function resets environment to the initial state
    def reset(self):
        self.current_distance = self.initial_distance
        return self.current_distance


class Agent:
    # an agent is initialized with the number of possible actions, learning rate (alpha), discount factor (gamma), and the exploration rate (epsilon)
    def __init__(self,environment, num_actions=4, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # agent's actions should include the type of club and the choice of power in the form of a tuple => (club-type,power-level)
        self.actions = [(club, round(power * 0.1, 1)) for club in ['woods', 'irons', 'hybrids', 'putter'] for power in range(10)]

        # Q-table for storing Q-values for each state-action pair
        self.q_table = {(state, action): 0 for state in range(1, environment.initial_distance + 1) for action in self.actions}



        # function to choose an action with epsilon-softpolicy. with a probability of epsilon, the agent will explore (choose a random action)
        # otherwise, it will exploit (choose the action with the highest estimated Q-value)

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = [self.q_table.get((state, action), 0) for action in self.actions]
            max_q_value = max(q_values)
            count = q_values.count(max_q_value)
            if count > 1:
                best_actions = [i for i in range(len(self.actions)) if q_values[i] == max_q_value]
                action = self.actions[np.random.choice(best_actions)]
            else:
                action = self.actions[np.argmax(q_values)]
        return action

    # update_q_table_sarsa function updates the Q-value for the current state-action pair based on the SARSA update rule
    # SARSA update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
    def update_q_table_sarsa(self, old_state, old_action, reward, new_state, new_action):
            old_q_value = self.q_table.get((old_state, old_action), 0)
            new_q_value = self.q_table.get((new_state, new_action), 0)
            self.q_table[(old_state, old_action)] = old_q_value + self.alpha * (
                    reward + self.gamma * new_q_value - old_q_value)

    # update Q-values based on the Q-learning algorithm
    # Q-learning update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * max[Q(S', a)] - Q(S, A)]
    def update_q_table_q_learning(self, old_state, action, reward, new_state):
        old_q_value = self.q_table.get((old_state, action), 0)
        # Q-Learning uses the maximum Q-value among new state's actions for the update, irrespective of the action chosen in the new state
        max_new_q_value = max([self.q_table.get((new_state, a), 0) for a in self.actions])
        self.q_table[(old_state, action)] = old_q_value + self.alpha * (
                reward + self.gamma * max_new_q_value - old_q_value)


class Experiment:
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment

    # function to run the SARSA algorithm
    def run_sarsa(self, max_episodes=20000):
        sarsa_scores = []
        sarsa_steps = []
        for episode in range(max_episodes):
            state = self.environment.reset()
            action = self.agent.get_action(state)
            done = False
            steps = 0
            total_reward = 0
            while not done:
                new_state, reward, done = self.environment.step(action)
                new_action = self.agent.get_action(new_state)
                total_reward += reward
                steps += 1
                # note: In SARSA, we use the new action chosen by following the current policy to update the Q-value
                self.agent.update_q_table_sarsa(state, action, reward, new_state, new_action)
                state = new_state
                action = new_action
            sarsa_scores.append(total_reward)
            sarsa_steps.append(steps)
        print(f"Average reward for SARSA: {np.mean(sarsa_scores)}, Average steps: {np.mean(sarsa_steps)}")

    def run_q_learning(self, max_episodes=20000):
        ql_scores = []
        ql_steps = []
        for episode in range(max_episodes):
            state = self.environment.reset()
            done = False
            steps = 0
            total_reward = 0
            while not done:
                # agent chooses an action based on the current state
                action = self.agent.get_action(state)

                # chosen action is performed in the environment
                new_state, reward, done = self.environment.step(action)
                total_reward += reward
                steps += 1
                self.agent.update_q_table_q_learning(state, action, reward, new_state)
                # current state is updated to the new state
                state = new_state
            ql_scores.append(total_reward)
            ql_steps.append(steps)
        print(f"\nAverage reward for Q-learning: {np.mean(ql_scores)}, Average steps: {np.mean(ql_steps)}")

    # print the optimal policy and Q-values
    def print_policy_and_q_values(self):
        optimal_policy = {}
        for state_action, q_value in self.agent.q_table.items():
            state, action = state_action
            if state not in optimal_policy or optimal_policy[state][1] < q_value:
                optimal_policy[state] = (action, q_value)

        sorted_policy = sorted(optimal_policy.items(), key=lambda x: x[0],reverse=True)  # Sort policy by distance

        print("Optimal Policy (State: [Action, Q-value]):")
        for state, action_q_value in sorted_policy:
            print(
                f"Distance {state}: Club {action_q_value[0][0]}, Power Level {action_q_value[0][1]}, Q-value {action_q_value[1]}")

        print("\n-------------------------------------------------------------\n")
        print("\nState-Action Value Function:")
        for state in range(1,self.environment.initial_distance + 1):
            print(f"Distance {state}:")
            for action in self.agent.actions:
                q_value = self.agent.q_table.get((state, action), 0)
                print(f"  Club {action[0]}, Power Level {action[1]}: Q-value {q_value}")



# initialize environment and agent
environment = GolfEnvironment()
agent = Agent(environment)

# run SARSA
experiment = Experiment(agent, environment)
experiment.run_sarsa()
print("\nSARSA Results:")
experiment.print_policy_and_q_values()

# run q-learning
agent = Agent(environment)
experiment = Experiment(agent, environment)
experiment.run_q_learning()
print("\nQ-Learning Results:")
experiment.print_policy_and_q_values()
