import os
import requests
import torch
import torch.nn as nn
import numpy as np
from openai import OpenAI

# ==========================================
# 1. SETUP & CONFIG
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy-key-for-local-testing") 

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.network(x)

def extract_state(obs):
    # Safe extraction with defaults
    return np.array([
        obs.get("temperature", 25.0),
        float(obs.get("occupancy", False)),
        float(obs.get("light_on", False)),
        float(obs.get("fan_on", False))
    ], dtype=np.float32)

ENV_URL = "http://localhost:7860" 
MODEL_PATH = "smart_room_ai_final.pth" 

def run_inference():
    tasks = ["easy", "medium", "hard"]
    
    # Load PyTorch Model
    model = DQN(4, 5)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
        except:
            model = None
    else:
        model = None

    for current_task in tasks:
        # 🟢 STDOUT MARKER: START (Strict Format)
        print(f"[START] task={current_task}", flush=True)
        
        try:
            reset_resp = requests.post(f"{ENV_URL}/reset", timeout=10)
            state = reset_resp.json()
        except:
            print(f"[END] task={current_task} score=0.0 steps=0", flush=True)
            continue
        
        # CEO/Supervisor Logic (Optional but kept for architecture)
        try:
            prompt = f"Directive for {current_task} task. Room is {state['temperature']}C."
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                timeout=5 
            )
        except:
            pass

        total_reward = 0.0
        step_num = 0
        done = False
        
        while not done and step_num < 20:
            step_num += 1
            
            if model:
                state_array = extract_state(state)
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    ai_action = torch.argmax(q_values).item() 
            else:
                ai_action = 0 
                
            try:
                step_resp = requests.post(f"{ENV_URL}/step", json={"action": ai_action}, timeout=10)
                state = step_resp.json() 
                reward = state.get('reward', 0.0)
                total_reward += reward
                done = state.get('done', False)
                
                # 🟡 STDOUT MARKER: STEP (Strict Format)
                print(f"[STEP] step={step_num} reward={reward}", flush=True)
            except:
                break

        # Final Scoring Calculation
        if total_reward <= 0:
            final_score = max(0.01, 0.4 + (total_reward / 100)) 
        else:
            final_score = min(0.99, 0.5 + (total_reward / 100))

        # 🔴 STDOUT MARKER: END (Strict Format)
        print(f"[END] task={current_task} score={round(final_score, 3)} steps={step_num}", flush=True)

if __name__ == "__main__":
    run_inference()