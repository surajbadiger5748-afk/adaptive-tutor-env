from copy import deepcopy
import uuid

from app.state import StudentState, EnvState
from app.student import StudentSimulator
from app.observation import build_observation
from app.reward import compute_reward

from app.types import Action, Observation, State


class AdaptiveTutorEnv:

    def __init__(self):
        self.simulator = StudentSimulator()
        self.state = None
        self.episode_id = str(uuid.uuid4())

    # ✅ reset
    def reset(self) -> Observation:
        student = StudentState()
        self.state = EnvState(student=student)
        self.state.initial_knowledge = student.knowledge

        self.episode_id = str(uuid.uuid4())

        obs_dict = build_observation(self.state)

        return Observation(
            turn=obs_dict["turn"],
            last_score=obs_dict["last_score"],
            history=obs_dict["history"]
        )

    # ✅ step
    def step(self, action: Action):
        action_dict = action.__dict__

        prev_state = deepcopy(self.state)
        student = self.state.student

        result = None

        if action_dict["type"] == "EXPLAIN":
            self.simulator.apply_explanation(student, action_dict)

        elif action_dict["type"] == "QUIZ":
            result, score = self.simulator.generate_quiz_response(student)
            self.state.last_score = score

        elif action_dict["type"] == "HINT":
            self.simulator.apply_hint(student)

        else:
            raise ValueError("Invalid action type")

        self.state.history.append({
            "action": action_dict,
            "result": result
        })

        self.state.turn += 1

        reward = compute_reward(self.state, action_dict, prev_state)

        done = False
        if self.state.turn >= self.state.max_turns:
            done = True
        if student.knowledge >= 0.9:
            done = True

        obs_dict = build_observation(self.state)

        obs = Observation(
            turn=obs_dict["turn"],
            last_score=obs_dict["last_score"],
            history=obs_dict["history"]
        )

        info = {
            "knowledge": student.knowledge
        }

        return obs, reward, done, info

    # ✅ FIXED: INSIDE CLASS
    def get_state(self) -> State:
        if self.state is None:
            return State(
                episode_id=self.episode_id,
                step_count=0
            )

        return State(
            episode_id=self.episode_id,
            step_count=getattr(self.state, "turn", 0)
        )