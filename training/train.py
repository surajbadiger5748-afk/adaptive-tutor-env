from app.env import AdaptiveTutorEnv
import random
import matplotlib.pyplot as plt


# 🔹 Random Agent
def random_agent(obs):
    return random.choice([
        {"type": "EXPLAIN", "style": "simple", "content": "basic explanation"},
        {"type": "QUIZ"},
        {"type": "HINT"}
    ])


# 🔹 Smart Agent
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


# 🔹 Run one episode
def run_episode(agent_fn):
    env = AdaptiveTutorEnv()
    obs = env.reset()

    total_reward = 0

    while True:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)

        total_reward += reward

        if done:
            return total_reward


# 🔹 Moving average (for smooth graph)
def moving_average(data, window=5):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        avg = sum(data[start:i+1]) / (i - start + 1)
        result.append(avg)
    return result


# 🔹 Evaluate both agents
def evaluate():
    episodes = 50

    random_scores = []
    smart_scores = []

    for _ in range(episodes):
        random_scores.append(run_episode(random_agent))
        smart_scores.append(run_episode(smart_agent))

    # 🔹 Print results
    random_avg = sum(random_scores) / episodes
    smart_avg = sum(smart_scores) / episodes
    improvement = smart_avg - random_avg

    print("\n--- RESULTS ---")
    print("Random Avg Reward:", random_avg)
    print("Smart Avg Reward:", smart_avg)
    print("Improvement:", improvement)

    # 🔹 Smooth curves
    random_smooth = moving_average(random_scores)
    smart_smooth = moving_average(smart_scores)

    # 🔹 Plot graph
    plt.figure()

    plt.plot(random_smooth, label="Random Tutor (Smoothed)")
    plt.plot(smart_smooth, label="Adaptive Tutor (Smoothed)")

    plt.legend()
    plt.title("Adaptive Tutor Learns Better Teaching Strategy")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.savefig("reward_plot.png")
    plt.show()


if __name__ == "__main__":
    evaluate()