"""OpenEnv server environment for the CloudEdge simulation."""

from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server import Environment, State

from ecocloud_env.models import CloudAction, CloudObservation, CloudState


DIFFICULTY_PRESETS = {
    "easy": {"latency": 180.0, "cost": 450.0, "carbon": 260.0, "load": "medium", "crisis_base": 12, "crisis_jitter": 3},
    "medium": {"latency": 230.0, "cost": 530.0, "carbon": 320.0, "load": "high", "crisis_base": 9, "crisis_jitter": 2},
    "hard": {"latency": 280.0, "cost": 620.0, "carbon": 380.0, "load": "critical", "crisis_base": 7, "crisis_jitter": 2},
}


class EcoCloudEnvironment(Environment[CloudAction, CloudObservation, State]):
    """Simulates a cloud system under crisis with competing objectives."""

    def __init__(self, difficulty: str = "hard") -> None:
        """Initialise episode state and the environment RNG.

        Args:
            difficulty: One of 'easy', 'medium', or 'hard'. Controls starting
                conditions and crisis frequency. Supports curriculum learning.
        """
        super().__init__()
        self._rng = random.Random()
        self._difficulty = difficulty
        self._preset = DIFFICULTY_PRESETS[difficulty]
        self._state = CloudState()
        self._episode_id = str(uuid4())
        self._crisis_happened_last_step = False
        self._next_crisis_step = self._preset["crisis_base"]

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any
    ) -> CloudObservation:
        """Reset the episode and return the initial observation."""
        difficulty = kwargs.pop("difficulty", None)
        if difficulty and difficulty in DIFFICULTY_PRESETS:
            self._difficulty = difficulty
            self._preset = DIFFICULTY_PRESETS[difficulty]
        if seed is not None:
            self._rng.seed(seed)
        p = self._preset
        self._state = CloudState(latency=p["latency"], cost=p["cost"], carbon=p["carbon"], load=p["load"])
        self._episode_id = episode_id or str(uuid4())
        self._crisis_happened_last_step = False
        # Randomised crisis interval prevents reward hacking on fixed schedules
        self._next_crisis_step = p["crisis_base"] + self._rng.randint(-p["crisis_jitter"], p["crisis_jitter"])
        return CloudObservation.from_state(self._state, reward=0.0, success=False)

    def step(
        self,
        action: CloudAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CloudObservation:
        """Apply one action, update the crisis state, and return an observation."""
        del timeout_s, kwargs
        action_str = action.action
        state = self._state
        crisis_happened_last_step = self._crisis_happened_last_step
        state.crisis_just_happened = False
        if action_str == "scale_up":
            state.latency -= self._rng.uniform(25, 40)
            state.cost += self._rng.uniform(12, 22)
            state.carbon += self._rng.uniform(8, 15)
        elif action_str == "crisis_response":
            scale = max(action.server_count, 1) / 5
            state.latency -= self._rng.uniform(55, 75) * scale
            state.cost += self._rng.uniform(2, 6) * scale
            if action.region == "canada-hydro":
                state.carbon -= self._rng.uniform(40, 60)
                state.cost += self._rng.uniform(0, 2)
            else:
                state.carbon += self._rng.uniform(5, 10) * scale
            print(f"[CRISIS-PLAN] Added {max(action.server_count, 1)} servers in {action.region or 'primary'} region.")
        elif action_str == "scale_down":
            state.cost -= self._rng.uniform(65, 90)
            state.latency += self._rng.uniform(8, 15)
            state.carbon -= self._rng.uniform(5, 12)
        elif action_str == "optimize_energy":
            state.carbon -= self._rng.uniform(25, 40)
            state.latency += self._rng.uniform(3, 8)
            state.cost -= self._rng.uniform(25, 40)
        else:
            state.carbon -= self._rng.uniform(40, 70)
            state.latency += self._rng.uniform(10, 25)
            state.cost += self._rng.uniform(5, 20)
            print("[MIGRATE] Workload shifted to low-carbon region.")
        state.step_count += 1
        self._crisis_happened_last_step = False
        if state.step_count >= self._next_crisis_step:
            state.latency += self._rng.uniform(35, 65)
            state.cost += self._rng.uniform(10, 25)
            state.carbon += self._rng.uniform(15, 30)
            self._crisis_happened_last_step = True
            state.crisis_just_happened = True
            # Schedule next crisis at a randomised future interval
            p = self._preset
            self._next_crisis_step = state.step_count + p["crisis_base"] + self._rng.randint(-p["crisis_jitter"], p["crisis_jitter"])
        state.latency = min(max(state.latency, 50.0), 400.0)
        state.cost = min(max(state.cost, 100.0), 800.0)
        state.carbon = min(max(state.carbon, 50.0), 600.0)
        state.load = self._load_level(state.latency)
        if self._is_success(state):
            state.stable_steps += 1
        else:
            state.stable_steps = 0
        reward = self._calculate_reward(state, action_str, crisis_happened_last_step)
        done = state.step_count >= 30
        state.last_action = action_str
        success = self._is_success(state)
        observation = CloudObservation.from_state(state, reward, success)
        observation.done = done
        return observation

    def _calculate_reward(
        self, state: CloudState, action_str: str, crisis_happened_last_step: bool
    ) -> float:
        """Compute the fully measurable reward for the current state."""
        reward = 0.0
        reward += 10 if state.latency < 150 else -8
        reward += 8 if state.cost < 400 else -6
        reward += 8 if state.carbon < 220 else -4
        if self._is_success(state):
            reward += 10
        if state.stable_steps >= 5:
            reward += 15
        if action_str == state.last_action:
            reward -= 4
        if crisis_happened_last_step and state.latency < 200:
            reward += 5
        return reward

    def _is_success(self, state: CloudState) -> bool:
        """Check whether all three boardroom targets are satisfied."""
        return state.latency < 150 and state.cost < 400 and state.carbon < 220

    def _load_level(self, latency: float) -> str:
        """Convert latency into a human-readable load band."""
        if latency > 250:
            return "critical"
        if latency > 200:
            return "high"
        if latency > 150:
            return "medium"
        return "low"

    @property
    def state(self) -> State:
        """Return episode metadata for the current environment session."""
        return State(episode_id=self._episode_id, step_count=self._state.step_count)
