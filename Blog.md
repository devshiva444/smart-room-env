# 🚀 Smart Room AI: A 5000-Episode Reinforcement Learning Journey

## 🏠 The Challenge
For the Meta x Scaler OpenEnv Hackathon, I built an automated Smart Room system. The goal was to train an agent that manages AC, Lights, and Fans to save energy while keeping humans comfortable.

## 🧠 3-Layer Multi-Agent Architecture
My system isn't just a simple script; it’s a hybrid architecture:
1. **RL Worker (DQN):** The brain that learns from the environment.
2. **LLM Supervisor:** The executive that double-checks actions for logic.
3. **Safety Engine:** The final layer that prevents any dangerous or wasteful actions.

## 📈 Training Evidence: The Great Recovery
Training this agent for **5000 episodes** was a rollercoaster. 
- **The Success:** Initially, the agent learned quickly, reaching a +40 reward.
- **The Crash (Catastrophic Forgetting):** Around Episode 2400, the agent "forgot" everything. It stopped turning on the AC to save energy costs, even when the room was 35°C! Rewards crashed to **-69.00**.
- **The Comeback:** I didn't stop. I tuned the exploration, and by Episode 5000, the agent recovered to a stable **+39.39 reward**, proving it mastered the balance between comfort and cost.

## 📜 Official Training Logs
| Episode | Train Avg Reward | Status |
| :--- | :--- | :--- |
| 50 | -25.14 | Learning... |
| 1800 | +39.93 | Peak Mastery |
| 2500 | -68.57 | Catastrophic Forgetting |
| 5000 | +29.76 | **Stable Recovery** |

## 🎥 Video Demo
[[YOUTUBE LINK](https://youtu.be/zgw5JvdPbjw)]

---
**Author:** Shivraj Selar - Final Year CSE Student.