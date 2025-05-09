# Golf Simulation with Q-Learning and SARSA (RL Final Project)

This project simulates a simplified golf environment where an agent learns to reach a target distance of zero (hole-in-one) from 100 meters. The agent is trained using **Q-Learning** and **SARSA**, two foundational reinforcement learning algorithms, and follows a soft ε-policy for balanced exploration.

---

## 🧠 Problem Description

- **State**: Current distance to goal (1–100 meters)
- **Actions**: A tuple of `(club_type, power_level)`  
  Clubs: `woods`, `irons`, `hybrids`, `putter`  
  Power levels: 0.0 to 0.9 (10 steps)  
  → Total of 40 actions

- **Environment Dynamics**:
  - Each club has an average base distance and standard deviation (precision)
  - A wind disturbance affects every shot, sampled from `N(0, 1)`
  - The resulting distance is calculated by:  
    `distance_hit = (base_distance ± precision) * power ± wind`  
  - Agent transitions to a new state based on how close the shot lands to zero

---

## 🏌️ Objective

Learn an optimal policy that leads the agent from any starting distance to the hole (`distance = 0`) in the least number of steps.

---

## 📊 Rewards

- +100 if the shot exactly lands the ball in the hole (state = 0)
- -1 for each step otherwise

---

## 🔁 Algorithms Used

### SARSA (On-policy)
- Learns from actual action taken in the next state
- Conservative updates, smooth convergence

### Q-Learning (Off-policy)
- Uses the greedy action in the next state for learning
- Faster but sometimes unstable or over-optimistic

**Common Parameters:**
- α = 0.5  
- γ = 0.9  
- ε = 0.1 (soft policy)

---

## 📋 Output and Results

- Both algorithms trained for **20,000 episodes**
- **SARSA** converged slightly faster in early stages  
- **Q-Learning** achieved higher long-term expected reward  
- Final policies showed distinct club and power choices per state

### Sample Policy (Q-Learning):
```
Distance 87: Club woods, Power Level 0.8, Q = 79.6
Distance 52: Club hybrid, Power Level 0.5, Q = 64.1
Distance 10: Club putter, Power Level 0.1, Q = 33.2
```

### Observations:
- Agent learns to favor **woods** at long distances and **putter** near the hole  
- Q-values reflect expected returns from each action in each state  
- Action selection becomes more stable after ~15,000 episodes

---

## 📁 Files

- `main.py`: Full Python implementation of environment setup, agents, training, and evaluation

---

## ⚠️ Notes

- No external RL libraries used; all logic implemented manually
- Results printed to terminal (no plot visualization)
- Developed as part of academic coursework for educational purposes
