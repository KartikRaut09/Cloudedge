"""Run one local CloudEdge episode without starting the server."""

from __future__ import annotations

import os
import sys

from ecocloud_env.agents import Boardroom
from ecocloud_env.learner import AdaptiveBoardroomPolicy, BoardroomQLearner
from ecocloud_env.models import CloudAction, CloudObservation
from ecocloud_env.server.environment import EcoCloudEnvironment

DEFAULT_DEMO_SEED = 1
POLICY_PATH = os.path.join(
    os.path.dirname(__file__), "ecocloud_env", "artifacts", "boardroom_q_policy.json"
)


def action_label(action: CloudAction | str) -> str:
    """Format actions for compact episode logging."""
    if isinstance(action, str):
        return action
    if action.action == "crisis_response":
        return f"{action.action}[{action.server_count}@{action.region}]"
    return action.action


def print_step(step_number: int, obs: CloudObservation, action: CloudAction | str) -> None:
    """Print a compact summary for one environment step."""
    print(
        f"step={step_number} action={action_label(action)} latency={obs.latency:.1f} "
        f"cost={obs.cost:.1f} carbon={obs.carbon:.1f} "
        f"reward={obs.last_reward:.1f} success={obs.success}"
    )


def run_episode(seed: int = DEFAULT_DEMO_SEED, mode: str = "auto") -> CloudObservation:
    """Run one full episode using the boardroom policy."""
    env = EcoCloudEnvironment()
    learner = BoardroomQLearner.load(POLICY_PATH) if mode in {"auto", "trained"} and os.path.exists(POLICY_PATH) else None
    controller = AdaptiveBoardroomPolicy(Boardroom(), learner) if learner is not None else Boardroom()
    last_obs: CloudObservation | None = None
    recent_actions: list[str] = []
    obs = env.reset(seed=seed)
    effective_mode = "trained" if learner is not None else "heuristic"
    print(f"[DEMO] Using seed={seed} mode={effective_mode}")
    print_step(0, obs, "reset")
    while not obs.done:
        if obs.crisis_just_happened:
            print("[CRISIS] Traffic spike! System destabilized.")
        if isinstance(controller, AdaptiveBoardroomPolicy):
            action, _ = controller.decide(obs, last_obs, recent_actions, training=False, verbose=True)
        else:
            action, _ = controller.decide(obs, last_obs, recent_actions, verbose=True)
        last_obs = obs
        obs = env.step(action)
        recent_actions.append(action.action)
        print_step(obs.step_count, obs, action)
    return obs


def main() -> None:
    """Execute one episode and print the final result."""
    mode = "auto"
    seed = DEFAULT_DEMO_SEED
    args = sys.argv[1:]
    if args and args[0] in {"auto", "heuristic", "trained"}:
        mode = args[0]
        args = args[1:]
    if args:
        seed = int(args[0])
    final_obs = run_episode(seed=seed, mode=mode)
    print(
        f"final_success={final_obs.success} stable_steps={final_obs.stable_steps} "
        f"last_reward={final_obs.last_reward:.1f}"
    )


if __name__ == "__main__":
    main()
