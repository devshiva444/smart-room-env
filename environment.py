from openenv.core.env_server import Environment, Action, Observation, State
from pydantic import Field
import random

class SmartRoomAction(Action):
    action: int = Field(..., description="0: None, 1: Light ON, 2: Light OFF, 3: Fan ON, 4: Fan OFF")

class SmartRoomObservation(Observation):
    temperature: float
    light_on: bool
    fan_on: bool
    occupancy: bool
    energy_used: float
    reward: float
    done: bool

class SmartRoomState(State):
    step_count: int

class SmartRoomEnvironment(Environment):
    def __init__(self):
        self.step_count = 0
        self.temperature = 25.0
        self.outside_temp = 30.0
        self.light_on = False
        self.fan_on = False
        self.occupancy = False
        self.energy_used = 0.0
        self.reset()

    def reset(self) -> SmartRoomObservation:
        self.temperature = random.uniform(22.0, 32.0)
        self.outside_temp = 30.0 
        self.light_on = False
        self.fan_on = False
        self.occupancy = random.choice([True, False])
        self.energy_used = 0.0
        self.step_count = 0
        return self._get_obs(0.0, False)

    @property
    def state(self) -> SmartRoomState:
        return SmartRoomState(step_count=self.step_count)

    def _get_obs(self, reward: float, done: bool) -> SmartRoomObservation:
        return SmartRoomObservation(
            temperature=round(self.temperature, 2),
            light_on=self.light_on,
            fan_on=self.fan_on,
            occupancy=self.occupancy,
            energy_used=round(self.energy_used, 2),
            reward=float(reward),
            done=bool(done)
        )

    def step(self, action: SmartRoomAction) -> SmartRoomObservation:
        self.step_count += 1
        
        # Action Logic
        if action.action == 1: self.light_on = True
        elif action.action == 2: self.light_on = False
        elif action.action == 3: self.fan_on = True
        elif action.action == 4: self.fan_on = False

        # Physics Simulation
        if self.fan_on:
            self.temperature -= 1.0
            self.energy_used += 0.5
        else:
            self.temperature += (self.outside_temp - self.temperature) * 0.1
        
        if self.light_on:
            self.energy_used += 0.1

        if random.random() < 0.1:
            self.occupancy = not self.occupancy

        # Reward Calculation
        reward = 0.0
        if self.occupancy:
            if 22 <= self.temperature <= 26: reward += 5.0
            else: reward -= 1.0
            
            if self.light_on: reward += 2.0 
            else: reward -= 2.0  
        else:
            if self.light_on: reward -= 1.5 
            if self.fan_on: reward -= 1.5

        done = self.step_count >= 20
        return self._get_obs(reward, done)