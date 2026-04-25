def build_observation(state):
    return {
        "turn": state.turn,
        "last_score": state.last_score,
        "history": state.history[-3:],  # last 3 only
    }