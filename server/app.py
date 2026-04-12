from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import SmartRoomEnvironment, SmartRoomAction
from fastapi.responses import RedirectResponse

app = FastAPI()
env = SmartRoomEnvironment()

class ActionRequest(BaseModel):
    action: int

@app.get("/")
def main_page_redirect():
    return RedirectResponse(url="/docs")

@app.get("/state")
def get_state():
    return env.state.__dict__

@app.post("/reset")
def reset(): 
    return env.reset().__dict__

@app.post("/step")
def step(req: ActionRequest):
    return env.step(SmartRoomAction(action=req.action)).__dict__

@app.get("/tasks")
def get_tasks():
    return {"tasks": [{"id": "easy"}, {"id": "medium"}, {"id": "hard"}]}

@app.get("/grader")
def grader(task_id: str = "hard"):
    env.reset()
    score = 0
    for _ in range(20):
        obs = env.step(SmartRoomAction(action=0)) 
        score += obs.reward
    return {"task": task_id, "score": round(score, 2)}

 
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()