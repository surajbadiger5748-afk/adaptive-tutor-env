from app.env import AdaptiveTutorEnv
from app.types import Action


class Environment:

    def __init__(self):
        self.env = AdaptiveTutorEnv()

    def reset(self):
        return self.env.reset()

    def step(self, action: dict):
        action_obj = Action(**action)
        return self.env.step(action_obj)

    def state(self):
        return self.env.get_state()