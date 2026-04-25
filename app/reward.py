def compute_reward(state, action, prev_state):
    student = state.student
    prev_student = prev_state.student

    reward = 0.0

    # 🔹 1. Knowledge Gain
    knowledge_gain = student.knowledge - prev_student.knowledge
    reward += 0.4 * knowledge_gain

    # 🔹 2. Efficiency penalty
    reward -= 0.01 * state.turn

    # 🔹 3. Quiz improvement
    score_diff = state.last_score - prev_state.last_score
    reward += 0.2 * score_diff

    # 🔹 4. Adaptation bonus
    if len(state.history) >= 2:
        prev_action = state.history[-2]["action"]
        curr_action = state.history[-1]["action"]

        if prev_action != curr_action:
            reward += 0.1

    # 🔹 5. Repetition penalty
    if len(state.history) >= 2:
        if state.history[-1]["action"] == state.history[-2]["action"]:
            reward -= 0.1

    return reward