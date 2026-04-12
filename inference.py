import os
import requests
import torch
import torch.nn as nn
import numpy as np
from openai import OpenAI

# ==========================================
# 1. SETUP & FIX: Dummy Token for Local Testing
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy-key-for-local-testing") 

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ==========================================
# 2. THE EMPLOYEE (PyTorch DQN Architecture)
# ==========================================
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
    return np.array([
        obs["temperature"],
        float(obs["occupancy"]),
        float(obs["light_on"]),
        float(obs["fan_on"])
    ], dtype=np.float32)

ENV_URL = "http://localhost:7860" 
MODEL_PATH = "smart_room_ai_final.pth" 

def run_inference():
    tasks = ["easy", "medium", "hard"]
    
    # === HIRE THE EMPLOYEE (Load Custom Model) ===
    print(f"🧠 Hiring Expert PyTorch Employee from {MODEL_PATH}...", flush=True)
    try:
        model = DQN(4, 5) 
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() 
        print("✅ Employee Ready for Duty!\n", flush=True)
    except Exception as e:
        print(f"❌ Error loading PyTorch model: {e}", flush=True)
        model = None

    for current_task in tasks:
        print(f"--- [START] task={current_task} ---", flush=True)
        
        # 1. Room Reset (Check current condition)
        try:
            reset_resp = requests.post(f"{ENV_URL}/reset", timeout=10)
            state = reset_resp.json()
        except Exception as e:
            print(f"[END] task={current_task} score=0.0 steps=0", flush=True)
            continue
        
        # ==========================================
        # THE CEO SPEAKS (OpenAI Strategy Call)
        # ==========================================
        try:
            prompt = f"You are the AI CEO. The room temperature is {state['temperature']}C and occupancy is {state['occupancy']}. Give a 1-sentence directive to your PyTorch control agent for task '{current_task}'."
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                timeout=5 
            )
            print(f"👔 [CEO DIRECTIVE]: {response.choices[0].message.content}", flush=True)
        except Exception as e:
            print(f"👔 [CEO DIRECTIVE]: (Offline Mode) Focus on comfort and energy efficiency.", flush=True)

        total_reward = 0.0
        step_num = 0
        done = False
        
        # ==========================================
        # EMPLOYEE WORKS (PyTorch executing 20 steps)
        # ==========================================
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
                
                print(f"[STEP] step={step_num} reward={reward}", flush=True)
            except Exception as e:
                print(f"[STEP] step={step_num} reward=0.0", flush=True)
                break

        # ==========================================
        # THE FIX: Calculate Actual Hackathon Score
        # ==========================================
        
        if total_reward <= 0:
            final_score = max(0.01, 0.4 + (total_reward / 100)) 
        else:
            final_score = min(0.99, 0.5 + (total_reward / 100))

        # ==========================================
        # THE CEO REVIEWS (OpenAI Report Call)
        # ==========================================
        try:
            review_prompt = f"The PyTorch agent completed the task with a score of {final_score}. Give a 1-sentence performance review."
            review_resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": review_prompt}],
                timeout=5 
            )
            print(f"📊 [CEO REVIEW]: {review_resp.choices[0].message.content}", flush=True)
        except Exception as e:
            pass 

        # --- HACKATHON BOT REQUIREMENT: END MARKER ---
        print(f"[END] task={current_task} score={round(final_score, 3)} steps={step_num}\n", flush=True)

if __name__ == "__main__":
    run_inference()