"""Training and evaluation utilities for CloudEdge."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

from .agents import Boardroom
from .learner import AdaptiveBoardroomPolicy, BoardroomQLearner
from .models import CloudObservation
from .server.environment import EcoCloudEnvironment


def unpack_step(result: object) -> tuple[CloudObservation, float, bool]:
    """Support local env step results across observation/result shapes."""
    obs = result.observation if hasattr(result, "observation") else result
    reward = (
        result.reward
        if hasattr(result, "reward") and result.reward is not None
        else obs.last_reward
    )
    done = result.done if hasattr(result, "done") else getattr(obs, "done", False)
    return obs, float(reward), bool(done)


def _curriculum_difficulty(episode: int, total: int) -> str:
    """Return difficulty tier based on training progress (curriculum learning).

    Split: 20% easy → 20% medium → 60% hard.
    The agent builds foundational skills on easier scenarios before
    spending the majority of training on the target hard difficulty.
    """
    progress = episode / total
    if progress < 0.20:
        return "easy"
    if progress < 0.40:
        return "medium"
    return "hard"


def train_policy(
    episodes: int = 60,
    seed_offset: int = 100,
) -> tuple[BoardroomQLearner, dict[str, list[float] | list[int]]]:
    """Train a Q-learner on top of the boardroom and return metrics."""
    learner = BoardroomQLearner()
    controller = AdaptiveBoardroomPolicy(Boardroom(), learner)
    env = EcoCloudEnvironment()
    metrics: dict[str, list[float] | list[int]] = {
        "episode_rewards": [],
        "episode_success": [],
        "episode_final_latency": [],
        "episode_final_cost": [],
        "episode_final_carbon": [],
        "episode_stable_steps": [],
        "eval_rewards": [],
        "eval_success": [],
        "eval_final_latency": [],
        "eval_final_cost": [],
        "eval_final_carbon": [],
        "migrate_count_per_episode": [],
        "crisis_response_count_per_episode": [],
        "ep1_step_rewards": [],
        "ep1_latencies": [],
        "ep1_costs": [],
        "eplast_step_rewards": [],
        "eplast_latencies": [],
        "eplast_costs": [],
    }
    best_eval_reward = float("-inf")
    best_q_table: dict[str, dict[str, float]] = {}
    best_episode = 1
    for episode_number in range(1, episodes + 1):
        difficulty = _curriculum_difficulty(episode_number, episodes)
        obs = env.reset(seed=seed_offset + episode_number, difficulty=difficulty)
        last_obs: CloudObservation | None = None
        recent_actions: list[str] = []
        total_reward = 0.0
        success_this_ep = False
        migrate_count = 0
        crisis_count = 0
        step_rewards: list[float] = []
        step_latencies: list[float] = []
        step_costs: list[float] = []
        while not obs.done:
            with redirect_stdout(StringIO()):
                action, _ = controller.decide(
                    obs,
                    last_obs,
                    recent_actions,
                    training=True,
                    verbose=False,
                )
                next_obs, reward, done = unpack_step(env.step(action))
            learner.update(obs, action.action, reward, next_obs, done)
            last_obs = obs
            obs = next_obs
            recent_actions.append(action.action)
            total_reward += reward
            migrate_count += 1 if action.action == "migrate_region" else 0
            crisis_count += 1 if action.action == "crisis_response" else 0
            step_rewards.append(reward)
            step_latencies.append(obs.latency)
            step_costs.append(obs.cost)
            success_this_ep = success_this_ep or obs.success
        learner.end_episode()
        metrics["episode_rewards"].append(total_reward)
        metrics["episode_success"].append(1 if success_this_ep else 0)
        metrics["episode_final_latency"].append(obs.latency)
        metrics["episode_final_cost"].append(obs.cost)
        metrics["episode_final_carbon"].append(obs.carbon)
        metrics["episode_stable_steps"].append(obs.stable_steps)
        metrics["migrate_count_per_episode"].append(migrate_count)
        metrics["crisis_response_count_per_episode"].append(crisis_count)
        eval_result = evaluate_policy(episodes=5, learner=learner, seed_offset=9000)
        metrics["eval_rewards"].append(sum(eval_result["episode_rewards"]) / 5)
        metrics["eval_success"].append(sum(eval_result["episode_success"]) / 5)
        metrics["eval_final_latency"].append(sum(eval_result["episode_final_latency"]) / 5)
        metrics["eval_final_cost"].append(sum(eval_result["episode_final_cost"]) / 5)
        metrics["eval_final_carbon"].append(sum(eval_result["episode_final_carbon"]) / 5)
        if metrics["eval_rewards"][-1] > best_eval_reward:
            best_eval_reward = metrics["eval_rewards"][-1]
            best_q_table = copy.deepcopy(learner.q_table)
            best_episode = episode_number
        if episode_number == 1:
            metrics["ep1_step_rewards"] = step_rewards
            metrics["ep1_latencies"] = step_latencies
            metrics["ep1_costs"] = step_costs
        if episode_number == episodes:
            metrics["eplast_step_rewards"] = step_rewards
            metrics["eplast_latencies"] = step_latencies
            metrics["eplast_costs"] = step_costs
        if episode_number % 10 == 0:
            recent_success = sum(metrics["eval_success"][-10:]) / 10 * 100
            recent_reward = sum(metrics["eval_rewards"][-10:]) / 10
            print(
                f"Episode {episode_number:3d} | Eval Reward (last 10): {recent_reward:6.1f} | "
                f"Eval Success (last 10): {recent_success:5.1f}%"
            )
    if best_q_table:
        learner.q_table = best_q_table
    metrics["best_checkpoint"] = [best_episode]
    metrics["best_eval_reward"] = [best_eval_reward]
    return learner, metrics


def evaluate_policy(
    episodes: int = 20,
    learner: BoardroomQLearner | None = None,
    seed_offset: int = 1000,
) -> dict[str, Any]:
    """Evaluate either the heuristic boardroom or the trained learner."""
    env = EcoCloudEnvironment()
    controller = AdaptiveBoardroomPolicy(Boardroom(), learner)
    results: dict[str, Any] = {
        "episode_rewards": [],
        "episode_success": [],
        "episode_final_latency": [],
        "episode_final_cost": [],
        "episode_final_carbon": [],
    }
    for episode_number in range(episodes):
        obs = env.reset(seed=seed_offset + episode_number)
        last_obs: CloudObservation | None = None
        recent_actions: list[str] = []
        total_reward = 0.0
        success_this_ep = False
        while not obs.done:
            with redirect_stdout(StringIO()):
                action, _ = controller.decide(
                    obs,
                    last_obs,
                    recent_actions,
                    training=False,
                    verbose=False,
                )
                next_obs, reward, _ = unpack_step(env.step(action))
            last_obs = obs
            obs = next_obs
            recent_actions.append(action.action)
            total_reward += reward
            success_this_ep = success_this_ep or obs.success
        results["episode_rewards"].append(total_reward)
        results["episode_success"].append(1 if success_this_ep else 0)
        results["episode_final_latency"].append(obs.latency)
        results["episode_final_cost"].append(obs.cost)
        results["episode_final_carbon"].append(obs.carbon)
    return results
