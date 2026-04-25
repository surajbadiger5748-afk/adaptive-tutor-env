from fastapi import FastAPI
from dataclasses import asdict
from pydantic import BaseModel

from openenv_env.env import Environment

app = FastAPI()
env = Environment()


class ActionRequest(BaseModel):
    type: str
    style: str | None = None
    content: str | None = None


@app.get("/")
def home():
    return {
        "message": "OpenEnv Adaptive Tutor Running 🚀"
    }


@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": asdict(obs)}


@app.post("/step")
def step(action: ActionRequest):
    obs, reward, done, info = env.step(action.dict())
    return {
        "observation": asdict(obs),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return {"state": asdict(env.state())}