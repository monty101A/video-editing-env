# VideoEditingEnv 🎬

A reinforcement learning environment for video editing tasks using OpenEnv framework.

## Description
VideoEditingEnv simulates real-world video editing workflows where an AI agent learns to perform editing tasks step by step — just like a beginner editor would!

## Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| easy | 🟢 Easy | 5 | Basic Trim - Remove unwanted footage |
| medium | 🟡 Medium | 8 | Keyframe Animation - Create motion graphics |
| hard | 🔴 Hard | 12 | Full Motion Graphics - Complete sequence |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| action_type | string | trim, keyframe, color, export |
| target | string | Which clip to apply action on |
| value | string | Optional value for action |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| done | bool | Is episode complete |
| reward | float | Reward for last action |
| feedback | string | AI feedback to agent |
| current_task | string | Current task name |
| steps_remaining | int | Steps left |
| hint | string | Hint for next action |

## Setup
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Baseline Inference
```bash
export HF_TOKEN=your_token
export MODEL_NAME=gpt-4o-mini
python inference.py
```

## Baseline Scores

| Task | Score |
|------|-------|
| easy | 0.75 |
| medium | 0.60 |
| hard | 0.45 |