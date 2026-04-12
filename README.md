---
title: Smart Room Automation OpenEnv
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_file: server.py 
pinned: false
---

# 🏠 Smart Room Automation: An OpenEnv RL Benchmark

## 🌟 Overview
Welcome to the **Smart Room Automation** project! This is a custom Reinforcement Learning (RL) environment built on the **OpenEnv** framework. The goal is to train an AI agent that intelligently manages a room's climate and lighting. 

Unlike simple toy games, this environment simulates **real-world physics**—including heat transfer and probabilistic occupancy—to challenge the agent's ability to balance human comfort with strict energy efficiency.

### 🧠 The X-Factor: Hierarchical AI Architecture
To solve this environment, this project implements a **Supervisor-Worker (CEO-Employee) Architecture**:
1. **The Supervisor (OpenAI LLM):** Fulfills the OpenEnv evaluation criteria by acting as the high-level strategist. It assesses the initial room state and generates plain-text directives.
2. **The Expert Worker (PyTorch DQN):** Because LLMs are too slow and prone to hallucination for 20-step real-time physics manipulation, the actual environment control is delegated to a custom, locally-trained **128-Neuron Deep Q-Network**. It executes MDP steps in milliseconds, achieving scores far beyond standard LLM capabilities.

---

## 🎯 The Problem
Modern buildings contribute significantly to global energy waste. Lights are left on in empty rooms, and HVAC systems run at full power regardless of actual need. 

**Our Mission:** To create a standardized environment where RL agents can be evaluated on their ability to minimize carbon footprints without compromising indoor living standards.

---

## 🏗️ System Architecture

### 🔹 Perception (Observation Space)
The agent receives a rich state vector every step:
- **Temperature (`float`):** Real-time room temp (affected by outside weather and fan status).
- **Occupancy (`bool`):** Detects if a human is present (using a probabilistic model).
- **Device States:** Status of `light_on` and `fan_on`.
- **Energy Meter:** Cumulative energy consumption.

### 🔹 Control (Action Space)
| Value | Action | Impact |
|:---:|:---|:---|
| **0** | No Action | Maintains current state. |
| **1/2** | Light ON/OFF | Affects visibility and minor energy use. |
| **3/4** | Fan ON/OFF | High energy use, but regulates temperature. |

---

## 🧪 Reward Engineering & Reshaping
Initially, standard penalties led to "Reward Hacking" (the agent kept everything off to avoid energy penalties). We applied **Reward Reshaping** to prioritize human comfort:
- **Maximum Comfort (+5.0):** Maintaining the "Sweet Spot" (22°C – 26°C) while occupied.
- **Utility (+2.0):** Correct lighting usage when someone is present.
- **Energy Waste Penalty (-1.5):** Penalty for devices running in an empty room (reduced to encourage exploration).

---

## 🧠 Evaluation Tasks
1. **Easy (Light Mastery):** Focus purely on occupancy-based lighting.
2. **Medium (Thermal Expert):** Focus on climate control and fan management.
3. **Hard (Eco-Optimizer):** Optimizing both systems simultaneously.

---

## 🛠️ Technical Setup & Execution

### 1. Local Installation
```bash
# Clone the repository
git clone [https://github.com/devshiva444/smart-room-env.git](https://github.com/devshiva444/smart-room-env.git)
cd smart-room-env

# Setup Environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate 

# Install Core Dependencies
pip install -r requirements.txt

2. Running the AI Inference (Evaluation)
To run the automated Hackathon Grader, execute the inference script. Ensure the environment variables are set (The script uses a safe offline fallback if keys are missing during local testing).

Bash
# Set variables (Windows CMD)
set API_BASE_URL=[https://api.openai.com/v1](https://api.openai.com/v1)
set MODEL_NAME=gpt-3.5-turbo
set HF_TOKEN=your_huggingface_token

# Run the Agent
python inference.py
3. Server & Docker Deployment
This project is fully containerized and optimized for Hugging Face Spaces.

Bash
# Run server locally via Uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Or build via Docker
docker build -t smart-room-automation .
docker run -p 7860:7860 smart-room-automation
🔌 API Reference
POST /reset: Initialize a new episode.

POST /step: Submit an action (0-4) and receive observation.

GET /state: Returns current internal state.

GET /tasks: Lists all available evaluation tasks.

GET /grader?task_id={id}: Triggers automated evaluation.

👤 Author
Shivraj Selar Computer Science & Engineering Student

Developed for the Scaler x Meta OpenEnv Hackathon 2026.


