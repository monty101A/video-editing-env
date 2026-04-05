from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class VideoEditingAction:
    action_type: str
    target: str
    value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoEditingObservation:
    done: bool
    reward: float
    feedback: str
    current_task: str
    steps_remaining: int
    hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoEditingState:
    episode_id: str
    step_count: int
    task_name: str
    current_score: float
    max_steps: int