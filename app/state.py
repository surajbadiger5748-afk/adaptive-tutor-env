from dataclasses import dataclass, field
import random


# 🔹 Hidden Student State
@dataclass
class StudentState:
    knowledge: float = field(default_factory=lambda: random.uniform(0.1, 0.4))
    confusion: float = field(default_factory=lambda: random.uniform(0.0, 0.3))
    learning_rate: float = field(default_factory=lambda: random.uniform(0.2, 0.5))
    misconceptions: set = field(default_factory=set)

    def clamp(self):
        self.knowledge = max(0.0, min(1.0, self.knowledge))
        self.confusion = max(0.0, min(1.0, self.confusion))


# 🔹 Environment State
@dataclass
class EnvState:
    student: StudentState
    turn: int = 0
    max_turns: int = 10

    history: list = field(default_factory=list)
    last_score: float = 0.0
    initial_knowledge: float = 0.0