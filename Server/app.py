from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VideoEditingAction, VideoEditingObservation, VideoEditingState
from environment import VideoEditingEnvironment

app = FastAPI(title="VideoEditingEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

envs = {
    "easy": VideoEditingEnvironment("easy"),
    "medium": VideoEditingEnvironment("medium"),
    "hard": VideoEditingEnvironment("hard"),
}

current_task = "easy"


class ActionRequest(BaseModel):
    action_type: str
    target: str
    value: Optional[str] = None
    task: Optional[str] = "easy"


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(data: dict = {}):
    global current_task
    task = data.get("task", "easy")
    if task in envs:
        current_task = task
    obs = envs[current_task].reset()
    return {
        "observation": obs.__dict__,
        "reward": 0.0,
        "done": False,
        "info": {"task": current_task}
    }


@app.post("/step")
def step(action: ActionRequest):
    global current_task
    if action.task in envs:
        current_task = action.task
    act = VideoEditingAction(
        action_type=action.action_type,
        target=action.target,
        value=action.value,
    )
    obs = envs[current_task].step(act)
    return {
        "observation": obs.__dict__,
        "reward": obs.reward,
        "done": obs.done,
        "info": {"task": current_task}
    }


@app.get("/state")
def state():
    return envs[current_task].state().__dict__


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "descriptions": {
            "easy": "Basic Trim - Remove unwanted footage",
            "medium": "Keyframe Animation - Create motion graphics",
            "hard": "Full Motion Graphics - Complete sequence"
        }
    }
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
    