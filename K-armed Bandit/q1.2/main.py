import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 1000
EPS = 0
DICE_PROBABILITIES = [0.17, 0.23, 0.37, 0.05, 0.12, 0.06]


class DiceNumber:
    def __init__(self, p):
        self.p = p
        self.current_action_value = 1  # An optimistic value is set as the initial action value of all actions
        self.N = 0
        self.action_values = []
        self.rewards=[]

    def get_reward_from_environment(self):
        if np.random.random() < self.p:
            return +1
        else:
            return 0

    def update_action_value(self,new_reward):
        self.rewards.append(new_reward)
        self.N += 1
        self.current_action_value = np.mean(self.rewards)


def experiment():
    dice_numbers = [DiceNumber(p) for p in DICE_PROBABILITIES]

    for i in range(NUM_TRIALS):
        if np.random.random() < EPS:
            j = np.random.randint(len(dice_numbers))
        else:
            j = np.argmax([dice_number.current_action_value for dice_number in dice_numbers])

        # get new reward
        new_reward = dice_numbers[j].get_reward_from_environment()

        # update dice_number action_value
        dice_numbers[j].update_action_value(new_reward)

        print("iteration: ", i + 1, " selected-dice-number: ", j + 1, " all action-values: ", [dice_number.current_action_value for dice_number in dice_numbers])


        # log all dice_numbers action_values for this iteration
        for dice_number in dice_numbers:
            dice_number.action_values.append(dice_number.current_action_value)

    j = np.argmax([dice_number.current_action_value for dice_number in dice_numbers])
    print("Optimal action(dice-number):", (j + 1), " with action_value: ", dice_numbers[j].current_action_value)

    # plot the results
    plt.ylabel("action_value")
    plt.xlabel("time_step")
    plt.title("Optimistic Initial Value")
    plt.plot(dice_numbers[0].action_values, label="1")
    plt.plot(dice_numbers[1].action_values, label="2")
    plt.plot(dice_numbers[2].action_values, label="3")
    plt.plot(dice_numbers[3].action_values, label="4")
    plt.plot(dice_numbers[4].action_values, label="5")
    plt.plot(dice_numbers[5].action_values, label="6")
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    experiment()
