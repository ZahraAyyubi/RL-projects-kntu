# K-Armed Bandit – Unfair Dice (RL Homework 1)

This project simulates a 6-armed bandit environment where the agent must learn the optimal action through repeated interaction. The die is **unfair**, and the reward is only given (+1) when the predicted action matches the true die outcome.

The environment is **stationary**, and the probability distribution of outcomes remains constant over time.

---

## 🧪 Implemented Methods

### `q1.1` – Epsilon-Greedy
- Implemented for ε = 1, 0.1, 0.01
- Evaluates how exploration rate impacts learning speed and accuracy
- Results show that a lower ε (e.g. 0.1) balances exploration and exploitation effectively

### `q1.2` – Optimistic Initial Values
- All Q-values initialized to a high value (e.g. 1) to encourage exploration early on
- Converges faster in some cases compared to epsilon-greedy
- Helps escape early suboptimal action loops

### `q1.3` – Upper Confidence Bound (UCB)
- Applies confidence-based bonus to guide action selection
- The UCB formula balances uncertainty and known value estimates
- Performs well without relying on ε or initial values

---

## 📊 Results Summary

- All methods were simulated for 1000 episodes
- Estimated action-values (Q) were plotted across steps
- UCB and ε = 0.1 performed best in balancing accuracy and speed of convergence
- High ε values (ε=1) lead to slow or noisy learning due to excessive exploration

---

## 📁 Folder Structure

Each subfolder contains:
- `main.py`: Python implementation of the respective method
- `.png`: Plots of Q-value convergence over time

---

## ⚠️ Notes

- No external RL libraries were used (pure NumPy/Matplotlib)
- All analysis is self-contained in the code and plots 
