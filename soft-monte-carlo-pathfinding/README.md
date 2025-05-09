# Probabilistic Pathfinding with Monte Carlo – Soft Policy

This project models a pathfinding agent that moves through a one-dimensional grid from `START` to one of two terminal states (`T1` or `T2`) using a soft-policy Monte Carlo approach.

---

## 📌 Problem Description

- The environment is linear, consisting of 8 states: `T1`, `A`, `START`, `B`, `C`, `D`, `E`, `T2`
- The agent starts at `START` and chooses between `left` and `right` actions
- All transitions are **deterministic**, except one:  
  From state `A`, taking action `left` leads to `T1` with 0.2 probability and to `B` with 0.8
- Rewards:
  - All transitions: -1
  - Terminal states (`T1`, `T2`): 0

---

## 🔁 Methodology

- Algorithm: **First-Visit Monte Carlo**  
- Policy type: **Soft-policy** with ε = 0.4  
- Action selection is probabilistic based on ε-greedy soft strategy
- State-action values `Q(s,a)` are estimated from returns
- Policy is improved iteratively from estimated Q-values
- Three values of γ were tested: **1.0, 0.5, 0.1**

---

## 📊 Output

- For each γ value, the learned policy is printed in terminal
- Observations:
  - γ = 1.0 → favors long-term reward, prefers `T2`
  - γ = 0.1 → short-sighted, prefers `T1` due to fewer steps
  - γ = 0.5 → mixed behavior

---

## 📁 File Structure

- `main.py`: Full Monte Carlo implementation with soft policy exploration

---

## ⚠️ Notes

- No external libraries were used (NumPy only)
- This implementation does not use any visualization; all results are shown in console
- The project was developed for academic exploration and experimentation
