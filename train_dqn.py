import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

BASE_URL = "http://127.0.0.1:7860"

# ==========================================
# 1. THE EXPERT LOGIC 
# ==========================================
def smart_policy(obs):
    temp = obs["temperature"]
    occupancy = obs["occupancy"]
    light_on = obs["light_on"]
    fan_on = obs["fan_on"]

    # Fan logic
    if temp > 26 and not fan_on: return 3  # fan ON
    if temp < 22 and fan_on: return 4      # fan OFF

    # Light logic
    if occupancy and not light_on: return 1  # light ON
    if not occupancy and light_on: return 2  # light OFF

    return 0  # do nothing

# ==========================================
# 2. THE BRAIN (Neural Network)
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

# ==========================================
# 3. THE RL AGENT
# ==========================================
class DQNAgent:
    def __init__(self):
        self.state_dim = 4      
        self.action_dim = 5     
        
        self.gamma = 0.99       
        self.epsilon = 1.0      
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        self.memory = deque(maxlen=5000) 
        
        self.model = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim) 
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item() 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
                
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor)
            target_f[0][action] = target 
            
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# 4. MEMORY SEEDING (The Masterstroke)
# ==========================================
def prefill_memory(agent, episodes=50):
    print(f"🧠 Step 1: Filling AI Memory with Expert Rules ({episodes} episodes)...")
    for _ in range(episodes):
        response = requests.post(f"{BASE_URL}/reset")
        obs = response.json()
        state = extract_state(obs)
        done = False
        
        while not done:
            action = smart_policy(obs)
            
            response = requests.post(f"{BASE_URL}/step", json={"action": action})
            next_obs = response.json()
            next_state = extract_state(next_obs)
            reward = next_obs["reward"]
            done = next_obs["done"]
            
            agent.remember(state, action, reward, next_state, done)
            obs = next_obs
            state = next_state
            
    print("✅ Memory filled! AI already knows the basics now.")

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_ai(episodes=300):
    agent = DQNAgent()
    
    prefill_memory(agent, episodes=50)
    
    print("🚀 Step 2: Starting Deep AI Training...")
    for e in range(episodes):
        response = requests.post(f"{BASE_URL}/reset")
        obs = response.json()
        state = extract_state(obs)
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state) 
            
            response = requests.post(f"{BASE_URL}/step", json={"action": action})
            next_obs = response.json()
            next_state = extract_state(next_obs)
            reward = next_obs["reward"]
            done = next_obs["done"]
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward

        print(f"Episode {e+1}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    print("🏆 Training Complete! Model is Super Smart now.")
    torch.save(agent.model.state_dict(), "smart_room_ai_final.pth")

if __name__ == "__main__":
   
    train_ai(episodes=500)

