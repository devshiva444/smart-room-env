from pydantic import BaseModel
from typing import List, Optional

class ActionRequest(BaseModel):
    action: int

class ObservationResponse(BaseModel):
    temperature: float
    light_on: bool
    fan_on: bool
    occupancy: bool
    energy_used: float
    reward: float
    done: bool

class TaskInfo(BaseModel):
    id: str
    name: str
    goal: str

class TasksResponse(BaseModel):
    tasks: List[TaskInfo]