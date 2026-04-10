import uuid
from models import VideoEditingAction, VideoEditingObservation, VideoEditingState

TASKS = {
    "easy": {
        "name": "Basic Trim",
        "description": "Trim a video clip to remove unwanted footage",
        "correct_actions": ["trim"],
        "max_steps": 5,
    },
    "medium": {
        "name": "Keyframe Animation",
        "description": "Add keyframes to create motion graphics",
        "correct_actions": ["keyframe", "trim"],
        "max_steps": 8,
    },
    "hard": {
        "name": "Full Motion Graphics",
        "description": "Create a complete motion graphics sequence",
        "correct_actions": ["keyframe", "trim", "color", "export"],
        "max_steps": 12,
    }
}


class VideoEditingEnvironment:

    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.task = TASKS[task_name]
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        self.completed_actions = []
        self.current_score = 0.0
        self.done = False

    def reset(self) -> VideoEditingObservation:
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        self.completed_actions = []
        self.current_score = 0.0
        self.done = False

        return VideoEditingObservation(
            done=False,
            reward=0.01,
            feedback=f"Task started: {self.task['description']}. Begin editing!",
            current_task=self.task["name"],
            steps_remaining=self.task["max_steps"],
            hint=f"Try action: {self.task['correct_actions'][0]}",
        )

    def step(self, action: VideoEditingAction) -> VideoEditingObservation:
        self.step_count += 1
        steps_remaining = self.task["max_steps"] - self.step_count
        reward = 0.01
        feedback = ""
        hint = None

        correct_actions = self.task["correct_actions"]

        if action.action_type in correct_actions:
            if action.action_type not in self.completed_actions:
                self.completed_actions.append(action.action_type)
                reward = 0.85 / len(correct_actions)
                self.current_score += reward
                feedback = f"✅ Great! '{action.action_type}' applied correctly!"
            else:
                reward = 0.02
                feedback = f"⚠️ '{action.action_type}' already done. Try next step!"
        else:
            reward = 0.02
            feedback = f"❌ '{action.action_type}' is not correct here. Think again!"

        # Check remaining hints
        remaining = [a for a in correct_actions if a not in self.completed_actions]
        if remaining:
            hint = f"Next try: {remaining[0]}"

        # Check if task complete
        all_done = all(a in self.completed_actions for a in correct_actions)
        out_of_steps = steps_remaining <= 0

        if all_done:
            self.done = True
            reward = min(0.95, reward + 0.4)
            self.current_score = min(0.95, self.current_score + 0.4)
            feedback = "🎉 Task Complete! All steps done perfectly!"
        elif out_of_steps:
            self.done = True
            feedback = "⏰ Out of steps! Task incomplete."

        return VideoEditingObservation(
            done=self.done,
            reward=round(max(0.01, min(0.95, reward)), 3),
            feedback=feedback,
            current_task=self.task["name"],
            steps_remaining=max(0, steps_remaining),
            hint=hint,
        )

    def state(self) -> VideoEditingState:
        return VideoEditingState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            task_name=self.task_name,
            current_score=round(max(0.01, min(0.95, self.current_score)), 3),
            max_steps=self.task["max_steps"],
        )