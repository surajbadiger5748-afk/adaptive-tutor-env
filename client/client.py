import requests

BASE_URL = "http://127.0.0.1:8000"


def reset():
    res = requests.post(f"{BASE_URL}/reset")
    return res.json()


def step(action):
    res = requests.post(f"{BASE_URL}/step", json=action)
    return res.json()


def get_state():
    res = requests.get(f"{BASE_URL}/state")
    return res.json()


if __name__ == "__main__":
    print("RESET:")
    print(reset())

    print("\nSTEP:")
    print(step({
        "type": "EXPLAIN",
        "style": "simple",
        "content": "Recursion example"
    }))

    print("\nSTATE:")
    print(get_state())