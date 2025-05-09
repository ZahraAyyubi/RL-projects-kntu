# Grid Navigation using Value Iteration

This project solves a **Markov Decision Process (MDP)** in a grid-based industrial environment using **Value Iteration**. The agent is a robot whose battery drops to 20% and must learn the best path to a charging station.

---

## 🧠 Problem Description

The environment is modeled as a 2D grid representing a factory floor. Each cell can be one of the following:

- `O`: An empty navigable cell (reward = 0)
- `X`: A wall/obstacle (non-accessible)
- `G`: The goal cell representing the charging station (reward = +10)

The robot can take one of four actions from each state: `up`, `down`, `left`, `right`.  
Transitions may be stochastic, but follow a fixed structure. The goal is to learn the **optimal value function** and derive a policy that leads the robot to the charging station from any initial cell.

---

## 🔁 Methodology

- The algorithm used is **Value Iteration**, a dynamic programming method for solving MDPs.
- State transitions and rewards are encoded in dictionaries.
- The value function is updated iteratively until convergence.
- The final policy is determined by selecting the action with the highest expected value at each state.

---

## ⚠️ Notes

- No external libraries like Gym or RLlib were used.
- Results were interpreted through terminal logs and value arrays.
- The code was designed for educational clarity, not production efficiency.
