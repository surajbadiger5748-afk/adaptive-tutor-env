from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# 🔹 Action: what agent does
@dataclass
class Action:
    type: str
    style: Optional[str] = None
    content: Optional[str] = None


# 🔹 Observation: what agent sees
@dataclass
class Observation:
    turn: int
    last_score: float
    history: List[Dict[str, Any]]


# 🔹 State: episode metadata
@dataclass
class State:
    episode_id: str
    step_count: int