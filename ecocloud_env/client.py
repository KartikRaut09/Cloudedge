"""Client helpers for connecting to the CloudEdge environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient, SyncEnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server import State

from .models import CloudAction, CloudObservation


class EcoCloudEnv(EnvClient[CloudAction, CloudObservation, State]):
    """Client for the CloudEdge OpenEnv environment."""

    action_class = CloudAction
    observation_class = CloudObservation

    def _step_payload(self, action: CloudAction) -> dict[str, Any]:
        """Serialise an action for the OpenEnv transport."""
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CloudObservation]:
        """Parse a reset or step result from the server."""
        return StepResult(
            observation=CloudObservation(**payload["observation"]),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        """Parse server-side episode metadata."""
        return State(**payload)

    async def reset(self, **kwargs: Any) -> StepResult[CloudObservation]:
        """Reset the environment and return the initial step result."""
        return await super().reset(**kwargs)

    async def step(
        self, action: CloudAction, **kwargs: Any
    ) -> StepResult[CloudObservation]:
        """Send an action and return the resulting step result."""
        return await super().step(action, **kwargs)


def make_env(base_url: str) -> SyncEnvClient:
    """Quick helper: returns a synchronous EcoCloudEnv client."""
    return EcoCloudEnv(base_url=base_url).sync()
