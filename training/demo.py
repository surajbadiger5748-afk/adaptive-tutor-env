from app.env import AdaptiveTutorEnv


def run_demo(agent_fn, label):
    print(f"\n===== {label} =====")

    env = AdaptiveTutorEnv()
    obs = env.reset()

    for step in range(6):
        action = agent_fn(obs)

        obs, reward, done, info = env.step(action)

        print(f"\nStep {step+1}")
        print("Action:", action)
        print("Reward:", round(reward, 4))
        print("Knowledge:", round(info["knowledge"], 4))

        if done:
            break


# 🔹 Random agent
import random
def random_agent(obs):
    return random.choice([
        {"type": "EXPLAIN", "style": "simple", "content": "basic explanation"},
        {"type": "QUIZ"},
        {"type": "HINT"}
    ])


# 🔹 Smart agent
def smart_agent(obs):
    if obs["turn"] == 0:
        return {
            "type": "EXPLAIN",
            "style": "simple",
            "content": "step by step example explanation"
        }

    if obs["last_score"] == 0:
        return {
            "type": "EXPLAIN",
            "style": "detailed",
            "content": "detailed explanation with example"
        }

    return {"type": "QUIZ"}


if __name__ == "__main__":
    run_demo(random_agent, "Random Tutor ❌")
    run_demo(smart_agent, "Adaptive Tutor ✅")