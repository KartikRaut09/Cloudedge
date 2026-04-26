"""Lightweight Q-learning policy layered on top of the EcoCloud boardroom."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .agents import Boardroom
from .models import CloudAction, CloudObservation

ACTIONS = ["scale_up", "scale_down", "optimize_energy", "migrate_region"]


class BoardroomQLearner:
    """Learns action preferences across episodes while keeping the boardroom prior."""

    def __init__(
        self,
        alpha: float = 0.22,
        gamma: float = 0.93,
        epsilon: float = 1.0,
        epsilon_min: float = 0.08,
        epsilon_decay: float = 0.965,
        boardroom_bonus: float = 0.75,
        seed: int = 7,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.boardroom_bonus = boardroom_bonus
        self.rng = random.Random(seed)
        self.q_table: dict[str, dict[str, float]] = {}

    def bucket_state(self, obs: CloudObservation) -> tuple[str, ...]:
        """Compress the continuous state into a small, learnable bucket."""
        latency_band = "critical" if obs.latency > 220 else "elevated" if obs.latency > 150 else "target"
        cost_band = "critical" if obs.cost > 550 else "high" if obs.cost > 400 else "target"
        carbon_band = "critical" if obs.carbon > 320 else "high" if obs.carbon > 220 else "target"
        recovery_band = "stable" if obs.stable_steps >= 2 else "recovering"
        crisis_flag = "crisis" if obs.crisis_just_happened else "steady"
        return (latency_band, cost_band, carbon_band, recovery_band, crisis_flag)

    def choose_action(
        self,
        obs: CloudObservation,
        boardroom_action: str | None = None,
        training: bool = False,
    ) -> CloudAction:
        """Pick an action with epsilon-greedy exploration and boardroom bias."""
        if obs.crisis_just_happened:
            return CloudAction(action="crisis_response", server_count=5, region="canada-hydro")
        if training and self.rng.random() < self.epsilon:
            return CloudAction(action=self.rng.choice(self._exploration_candidates(obs, boardroom_action)))
        state_key = self._state_key(obs)
        values = self._ensure_state(state_key)
        scored_actions = {}
        for action in ACTIONS:
            score = values[action]
            if action == boardroom_action:
                score += self.boardroom_bonus
            score += self._safety_bias(obs, action)
            scored_actions[action] = score
        best_action = max(ACTIONS, key=lambda action: (scored_actions[action], -ACTIONS.index(action)))
        return CloudAction(action=best_action)

    def update(
        self,
        obs: CloudObservation,
        action: str,
        reward: float,
        next_obs: CloudObservation,
        done: bool,
    ) -> None:
        """Apply a Q-learning update from one transition."""
        if action not in ACTIONS:
            return
        state_key = self._state_key(obs)
        next_key = self._state_key(next_obs)
        state_values = self._ensure_state(state_key)
        next_values = self._ensure_state(next_key)
        clipped_reward = max(min(reward, 50.0), -50.0)
        target = clipped_reward if done else clipped_reward + self.gamma * max(next_values.values())
        state_values[action] += self.alpha * (target - state_values[action])

    def end_episode(self) -> None:
        """Decay exploration after each training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Persist learner weights for demo and evaluation reuse."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "boardroom_bonus": self.boardroom_bonus,
            "q_table": self.q_table,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BoardroomQLearner":
        """Load a persisted learner from disk."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        learner = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            boardroom_bonus=payload["boardroom_bonus"],
        )
        learner.q_table = {
            key: {action: float(value) for action, value in values.items()}
            for key, values in payload["q_table"].items()
        }
        return learner

    def _ensure_state(self, state_key: str) -> dict[str, float]:
        """Lazily initialise a state's action values."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in ACTIONS}
        return self.q_table[state_key]

    def _state_key(self, obs: CloudObservation) -> str:
        """Encode the bucketed state into a JSON-safe string key."""
        return "|".join(self.bucket_state(obs))

    def _exploration_candidates(
        self, obs: CloudObservation, boardroom_action: str | None
    ) -> list[str]:
        """Bias random exploration toward actions that are still directionally sane."""
        candidates = list(ACTIONS)
        if obs.latency > 210:
            candidates = [action for action in candidates if action != "scale_down"]
        if obs.carbon < 220 and "migrate_region" in candidates:
            candidates.remove("migrate_region")
        if obs.cost > 560 and "scale_up" in candidates and obs.latency < 170:
            candidates.remove("scale_up")
        if boardroom_action in ACTIONS and boardroom_action not in candidates:
            candidates.append(boardroom_action)
        return candidates or list(ACTIONS)

    def _safety_bias(self, obs: CloudObservation, action: str) -> float:
        """Apply light guardrails so exploration does not destroy recoverability."""
        bias = 0.0
        if obs.latency > 190 and action == "scale_up":
            bias += 0.8
        if obs.latency > 170 and action in {"scale_down", "optimize_energy"}:
            bias -= 0.8
        if obs.cost > 520 and obs.latency <= 150 and action == "scale_down":
            bias += 0.7
        if obs.carbon > 280 and obs.latency <= 165 and action == "migrate_region":
            bias += 0.8
        if obs.carbon > 220 and action == "optimize_energy":
            bias += 0.45
        if obs.carbon < 220 and action == "migrate_region":
            bias -= 0.6
        if obs.cost > 500 and action == "scale_up" and obs.latency <= 160:
            bias -= 0.75
        return bias


class AdaptiveBoardroomPolicy:
    """Combines heuristic boardroom reasoning with the learned Q-policy."""

    def __init__(self, boardroom: Boardroom | None = None, learner: BoardroomQLearner | None = None) -> None:
        self.boardroom = boardroom or Boardroom()
        self.learner = learner

    def decide(
        self,
        obs: CloudObservation,
        last_obs: CloudObservation | None = None,
        recent_actions: list[str] | None = None,
        training: bool = False,
        verbose: bool = True,
    ) -> tuple[CloudAction, list[str]]:
        """Return the heuristic action or a learner-adjusted override."""
        heuristic_action, log = self.boardroom.decide(
            obs, last_obs, recent_actions, verbose=verbose
        )
        if self.learner is None:
            return heuristic_action, log
        if obs.crisis_just_happened or self._is_forced_override(log):
            return heuristic_action, log
        learned_action = self.learner.choose_action(
            obs,
            boardroom_action=heuristic_action.action,
            training=training,
        )
        if learned_action.action == heuristic_action.action:
            return learned_action, log
        override_line = (
            f"[LEARNER] Override: {heuristic_action.action} -> {learned_action.action}"
        )
        log = list(log) + [override_line]
        if verbose:
            print(override_line)
            print("-" * 50)
        return learned_action, log

    @staticmethod
    def _is_forced_override(log: list[str]) -> bool:
        """Keep hard-coded crisis and anti-oscillation overrides intact."""
        return any(
            "cycle detected" in line.lower() or "oscillation detected" in line.lower()
            for line in log
        )


def learner_payload(learner: BoardroomQLearner) -> dict[str, Any]:
    """Expose a compact learner summary for reporting."""
    return {
        "states_learned": len(learner.q_table),
        "epsilon": round(learner.epsilon, 4),
        "alpha": learner.alpha,
        "gamma": learner.gamma,
    }
