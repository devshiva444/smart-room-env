---
title: Multi-Agent Smart Room System - OpenEnv
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py 
pinned: true
---

# 🤖 3-Layer Multi-Agent Smart Room System
**An OpenEnv RL + LLM Hybrid — Meta x Scaler OpenEnv Hackathon 2026**

## Quick demo
Add your unlisted demo link here (2 minutes): [[YOUTUBE LINK](https://youtu.be/zgw5JvdPbjw)]

---

## What this project does (short)
- Learns to control room devices (lights, fan, AC) from a 7‑dim state vector.
- Optimizes for human comfort while minimizing energy use and enforcing safety rules.
- Uses a 3-layer pipeline: a DQN proposes actions, an LLM reviews/adjusts them, and a safety engine enforces hard constraints.

---

## Key Features
- Multi-Agent 3-layer control: `RL Worker` + `LLM Supervisor` + `Safety Engine`.
- Trained DQN (7 → 128 → 128 → 9) with 5000 episodes of curriculum learning.
- Human-readable training evidence in `Training_Evidence.ipynb` (V-shaped learning + recovery).
- FastAPI dashboard for live simulation and evaluation (`/grader`, `/state`, `/ai_step`).
- Docker-ready for Hugging Face Spaces (port 7860).

---

## Goals & Impact
- Primary goal: Maintain occupant comfort while cutting unnecessary energy use.
- Secondary goals: Safe, interpretable actions (LLM explanations), reproducible training evidence for judges.
- Intended impact: Reduce wasteful HVAC runtime and avoid comfort violations during extreme conditions.

---

## Energy & Waste — how we measure impact
This repository includes simulation-based measurements (see `Training_Evidence.ipynb`) rather than field data. Below are measured example metrics taken from the 5000-episode training evidence (open the notebook for full plots and per-scenario tables):

- Training performance highlights:
	- Peak average training reward: **+41.08** (Episode 1300)
	- Lowest average training reward (catastrophic forgetting): **-69.04** (Episode 2500)
	- Final stabilized average reward: **+39.39** (Episode 5000)
	- Total reward swing (valley → final): **~110.12** points (recovery magnitude)

- Representative action energy costs (simulation units):

	| Action | Energy Cost (units) |
	|--------|---------------------|
	| Light ON | 0.1 |
	| Light OFF | 0 |
	| Fan Speed 1 | 0.2 |
	| Fan Speed 2 | 0.4 |
	| Fan Speed 3 | 0.6 |
	| Fan OFF | 0 |
	| AC ON | 1.0 |
	| AC OFF | 0 |

- What the notebook shows:
	- Per-episode energy consumption traces for agent vs baselines (always-on AC, naive thermostat).
	- Scenario-based percent reductions (e.g., simulated heatwave/day/night) summarized as bar charts and a short table.

For exact per-scenario percent savings and the comparison table, run the `Energy Comparison` cell in `Training_Evidence.ipynb`. The notebook contains code to reproduce these numbers and the plots used in the hackathon submission.

---

## How it works (technical)
- Input (state): [temperature, occupancy, light, fan, AC, time_of_day, sleep_mode]
- Action space: 9 discrete actions (Do nothing, Light ON/OFF, Fan SPEED1-3/OFF, AC ON/OFF).
- Reward: multi-component (comfort penalty, energy penalty, safety penalty, stability bonus).
- RL algorithm: DQN with experience replay, target-network updates, epsilon-greedy exploration.

Architecture summary:

1. RL Worker: DQN proposes best action from state vector.
2. LLM Supervisor: Reviews action and can suggest safer/clearer alternatives.
3. Safety Engine: Enforces hard constraints (e.g., do not enable AC below X°C when window open).

---

## Quick Start (local)
1. Create virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Run dashboard (dev):

```bash
python app.py
# Open http://localhost:7860
```

3. Run evaluation / inference:

```bash
python inference.py
```

4. Optional: build Docker (for Hugging Face Spaces):

```bash
docker build -t smart-room-ai .
docker run -p 7860:7860 -e HF_TOKEN="your_token" smart-room-ai
```

---

## Files of interest
- `Training_Evidence.ipynb` — full 5000-episode logs, plots, architecture, and measured comparisons.
- `smart_room_ai_final.pth` — trained DQN weights (load with `torch.load`).
- `server/app.py` — FastAPI dashboard + endpoints.
- `core/multi_agent.py` — DQN model and agent wrapper.
- `core/llm_planner.py` — LLM supervisor logic.

---

## Run the energy comparison quickly
Open `Training_Evidence.ipynb` and run the cell labeled "Energy Comparison" — it will generate per-scenario bar charts and a short table with percent energy reduction vs baseline.

---

## Author
Shivraj Selar — Final-year CSE Student. Meta x Scaler OpenEnv Hackathon 2026.

---

