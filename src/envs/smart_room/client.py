import requests

class SmartRoomClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860"): 
        self.base_url = base_url

    def reset(self):
        response = requests.post(f"{self.base_url}/reset")
        return response.json()

    def step(self, action: int):
        response = requests.post(
            f"{self.base_url}/step", 
            json={"action": action}
        )
        return response.json()

    def get_state(self):
        response = requests.get(f"{self.base_url}/state")
        return response.json()

    def get_grader_score(self, task_id: str = "hard"):
        response = requests.get(f"{self.base_url}/grader", params={"task_id": task_id})
        return response.json()