import random


class StudentSimulator:

    def apply_explanation(self, student, action):
        quality = self.evaluate_explanation(action)

        knowledge_gain = (
            quality
            * student.learning_rate
            * (1 - student.confusion)
        )

        noise = random.uniform(-0.05, 0.05)
        knowledge_gain += noise

        student.knowledge += knowledge_gain

        if action.get("style") == "detailed" and student.knowledge < 0.3:
            student.confusion += 0.1

        student.clamp()

        return knowledge_gain

    def evaluate_explanation(self, action):
        content = action.get("content", "").lower()

        score = 0.5

        if "example" in content:
            score += 0.2

        if "step" in content:
            score += 0.2

        if len(content) < 20:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def generate_quiz_response(self, student):
        prob_correct = student.knowledge - student.confusion
        prob_correct = max(0.0, min(1.0, prob_correct))

        if random.random() < prob_correct:
            return "correct", 1.0
        else:
            return "incorrect", 0.0

    def apply_hint(self, student):
        gain = 0.05 * student.learning_rate
        student.knowledge += gain
        student.clamp()
        return gain