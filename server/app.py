from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import asdict


from app.env import AdaptiveTutorEnv
from app.types import Action

app = FastAPI()

env = AdaptiveTutorEnv()


# 🔹 Home route
@app.get("/")
def home():
    return {
        "message": "Adaptive Tutor Environment Running 🚀",
        "docs": "/docs",
        "endpoints": ["POST /reset", "POST /step", "GET /state"]
    }


# 🔹 Request model (API layer)
class ActionRequest(BaseModel):
    type: str
    style: str | None = None
    content: str | None = None


# 🔹 Reset endpoint
@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": asdict(obs)}  # convert dataclass → dict


# 🔹 Step endpoint
@app.post("/step")
def step(action: ActionRequest):
    # convert request → Action dataclass
    action_obj = Action(**action.dict())

    obs, reward, done, info = env.step(action_obj)

    return {
        "observation": asdict(obs),  # dataclass → dict
        "reward": reward,
        "done": done,
        "info": info
    }


# 🔹 State endpoint (useful for debugging / OpenEnv)

@app.get("/state")
def state():
    try:
        s = env.get_state()
        return {"state": asdict(s)}
    except Exception as e:
        return {"error": str(e)}