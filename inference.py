import asyncio
import os
import requests
from typing import List

# Config
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5
TASKS = ["easy", "medium", "hard"]

from openai import OpenAI

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward} done={done} error={error}", flush=True)


def log_end(success, steps, score, rewards):
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)


def get_model_action(observation: dict, task: str, step: int) -> dict:
    feedback = observation.get("feedback", "")
    hint = observation.get("hint", "")
    current_task = observation.get("current_task", "")

    prompt = f"""You are a video editing AI agent.
Current task: {current_task}
Feedback: {feedback}
Hint: {hint}
Step: {step}

Choose ONE action from: trim, keyframe, color, export

Respond in this exact format:
ACTION: <action_type>
TARGET: <clip1>
VALUE: <none>"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()

        action_type = "trim"
        target = "clip1"
        value = None

        for line in text.split("\n"):
            if line.startswith("ACTION:"):
                action_type = line.split(":")[1].strip().lower()
            elif line.startswith("TARGET:"):
                target = line.split(":")[1].strip()
            elif line.startswith("VALUE:"):
                val = line.split(":")[1].strip()
                value = None if val == "none" else val

        return {"action_type": action_type, "target": target, "value": value, "task": task}

    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        actions = ["trim", "keyframe", "color", "export"]
        return {"action_type": actions[step % len(actions)], "target": "clip1", "value": None, "task": task}


def run_task(task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env="VideoEditingEnv", model=MODEL_NAME)

    try:
        # Reset
        res = requests.post(f"{ENV_URL}/reset", json={"task": task_name})
        result = res.json()
        observation = result["observation"]
        done = result["done"]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_model_action(observation, task_name, step)

            res = requests.post(f"{ENV_URL}/step", json=action)
            result = res.json()

            observation = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action["action_type"], reward=reward, done=done, error=error)

            if done:
                break

        max_reward = MAX_STEPS * 1.0
        score = sum(rewards) / max_reward if max_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    print(f"[DEBUG] Starting VideoEditingEnv inference", flush=True)
    print(f"[DEBUG] ENV_URL: {ENV_URL}", flush=True)
    print(f"[DEBUG] MODEL: {MODEL_NAME}", flush=True)

    all_scores = []
    for task in TASKS:
        print(f"\n[DEBUG] Running task: {task}", flush=True)
        score = run_task(task)
        all_scores.append(score)
        print(f"[DEBUG] Task {task} score: {score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores)
    print(f"\n[DEBUG] Average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()