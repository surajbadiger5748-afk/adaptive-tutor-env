from app.env import AdaptiveTutorEnv

env = AdaptiveTutorEnv()

obs = env.reset()
print("Initial Observation:", obs)

actions = [
    {"type": "EXPLAIN", "style": "simple", "content": "Recursion example step by step"},
    {"type": "QUIZ"},
    {"type": "HINT"},
    {"type": "QUIZ"},
    {"type": "EXPLAIN", "style": "detailed", "content": "Detailed recursion explanation with example"},
    {"type": "QUIZ"}
]

for action in actions:
    obs, reward, done, info = env.step(action)

    print("\nAction:", action)
    print("Observation:", obs)
    print("Reward:", round(reward, 4))
    print("Done:", done)
    print("Knowledge:", round(info["knowledge"], 4))

    if done:
        break