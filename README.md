# 🧠 Adaptive Tutor Environment (OpenEnv + RL)

## 🚨 Problem

Current LLM tutors fail at **adaptive teaching**.

They:

* Over-explain to experts
* Under-explain to beginners
* Do not adapt based on feedback

Teaching is inherently **interactive and dynamic**, but most AI systems are static.

---

## 💡 Our Solution

We built an **OpenEnv-based reinforcement learning environment** where an AI learns how to teach effectively.

* A Tutor agent interacts with a simulated Student
* The Student has hidden internal state (knowledge, confusion, learning rate)
* The Tutor must infer student understanding through interaction
* Learning is driven by a **reward function**

---

## 🎯 Hackathon Theme Alignment

* ✅ **Long-Horizon Planning** → multi-step teaching interactions
* ✅ **World Modeling** → hidden student state inference
* ✅ **Self-Improving Systems** → reward-driven adaptation

---

## 🧩 Environment Design

### 🔹 Hidden Student State

* Knowledge (0–1)
* Confusion
* Learning rate

### 🔹 Observable State (Agent sees)

* Last quiz result
* Interaction history
* Current turn

👉 The agent must **infer student understanding indirectly**

---

## 🎮 Actions

* `EXPLAIN` → teach concept
* `QUIZ` → test understanding
* `HINT` → provide guidance

---

## 🏆 Reward Function

Multi-objective reward:

* 📈 Knowledge gain
* ⏱ Efficiency (fewer steps)
* 🧠 Quiz improvement
* 🔄 Adaptation bonus
* 🚫 Repetition penalty

👉 Prevents reward hacking and encourages **effective teaching strategies**

---

## 🚀 Live Demo

👉 Hugging Face Space:
https://huggingface.co/spaces/SurajBadiger/adaptive-tutor-env

---

## 🧪 Training

👉 Colab Notebook:
https://colab.research.google.com/drive/1C1NjA3I4lEkyihhdw5o6K9MkUxNkpkzh?usp=sharing

We simulate agent learning using reward signals from the environment.

The agent interacts with the environment over multiple steps and improves its teaching strategy.

---

## 📊 Results

### Reward over Time

![Reward Graph](reward_plot.png)

### Knowledge Growth

![Knowledge Graph](knowledge_plot.png)

👉 The adaptive tutor achieves **higher and more stable rewards**, showing learning behavior.

---

## ⚙️ Tech Stack

* FastAPI (Environment API)
* OpenEnv-style architecture
* Hugging Face Spaces (deployment)
* Python (simulation + logic)
* Matplotlib (evaluation)

---

## 🧠 Key Insight

Teaching is not just answering—it’s **adapting to the learner**.

Our environment enables AI systems to **learn how to teach**, not just respond.

---

## 📦 Submission Links

* 🔗 Hugging Face Space:
  https://huggingface.co/spaces/SurajBadiger/adaptive-tutor-env

* 🔗 Colab Notebook:
  https://colab.research.google.com/drive/1C1NjA3I4lEkyihhdw5o6K9MkUxNkpkzh?usp=sharing

* 🔗 GitHub Repository:
  **[ADD YOUR GITHUB LINK HERE]**

* 🔗 Demo Video:
  **[ADD YOUR VIDEO LINK HERE]**

---

## ⚠️ Notes for Reviewers

* The environment follows **OpenEnv principles** (`reset`, `step`, `state`)
* Fully deployed and runnable via Hugging Face Space
* Training is demonstrated through **reward-driven interaction loops**
* Designed to simulate **real-world adaptive teaching scenarios**

---
