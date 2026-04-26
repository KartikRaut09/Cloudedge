"""Pydantic models for CloudEdge state, actions, and observations."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from openenv.core.env_server import Action, Observation

VALID_ACTIONS = {"scale_up", "scale_down", "optimize_energy", "migrate_region", "crisis_response"}


class CloudState(BaseModel):
    """Tracks the internal cloud system state for one episode."""

    latency: float = 280.0
    cost: float = 620.0
    carbon: float = 380.0
    load: str = "critical"
    step_count: int = 0
    stable_steps: int = 0
    crisis_just_happened: bool = False
    last_action: str = ""


class CloudAction(Action):
    """Represents a single boardroom action for the environment."""

    action: str
    server_count: int = 0
    region: str = ""

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        """Reject unsupported environment actions."""
        if value not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {sorted(VALID_ACTIONS)}")
        return value

    @field_validator("server_count")
    @classmethod
    def validate_server_count(cls, value: int) -> int:
        """Reject invalid crisis scaling counts."""
        if value < 0:
            raise ValueError("server_count must be non-negative")
        return value


class CloudObservation(Observation):
    """Exposes the measurable cloud state returned after each step."""

    latency: float = 280.0
    cost: float = 620.0
    carbon: float = 380.0
    load: str = "critical"
    step_count: int = 0
    stable_steps: int = 0
    crisis_just_happened: bool = False
    last_action: str = ""
    last_reward: float = 0.0
    success: bool = False

    @classmethod
    def from_state(
        cls, state: CloudState, reward: float, success: bool
    ) -> "CloudObservation":
        """Build an observation snapshot from the current state."""
        return cls(
            **state.model_dump(),
            last_reward=reward,
            reward=reward,
            success=success,
        )
