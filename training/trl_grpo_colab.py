"""Colab-ready GRPO training script for EcoCloud War Room.

Run this on Colab / HF compute with:
    pip install trl transformers datasets accelerate peft bitsandbytes
    pip install -e .
    python training/trl_grpo_colab.py

This follows the Hugging Face TRL OpenEnv pattern:
https://huggingface.co/docs/trl/openenv
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from ecocloud_env.models import CloudAction
from ecocloud_env.server.environment import EcoCloudEnvironment

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/ecocloud-grpo"
TRAIN_PROMPTS = 256


SYSTEM_PROMPT = (
    "You are the EcoCloud War Room controller.\n"
    "Your mission: recover a cloud platform under crisis.\n"
    "Three metrics must ALL reach their targets simultaneously:\n"
    "  - latency < 150ms\n"
    "  - cost < $400/hr\n"
    "  - carbon < 220 units\n\n"
    "Available actions (respond with ONLY the action name):\n"
    "  scale_up - reduces latency but increases cost and carbon\n"
    "  scale_down - reduces cost but increases latency\n"
    "  optimize_energy - reduces carbon and cost but slightly increases latency\n"
    "  migrate_region - significantly reduces carbon but increases latency and cost\n\n"
    "Respond with exactly one action name per turn."
)


def make_env_and_run(action_name: str, env: EcoCloudEnvironment) -> tuple[float, dict]:
    """Execute a single action on the environment and return reward + state."""
    valid_actions = {"scale_up", "scale_down", "optimize_energy", "migrate_region"}
    if action_name not in valid_actions:
        return -20.0, {}  # penalty for invalid action
    obs = env.step(CloudAction(action=action_name))
    return float(obs.last_reward), {
        "latency": obs.latency,
        "cost": obs.cost,
        "carbon": obs.carbon,
        "success": obs.success,
    }


def extract_action(text: str) -> str:
    """Parse the model's output to extract an action name."""
    # Normalize: lowercase, replace spaces/hyphens with underscores
    text = text.strip().lower().replace(" ", "_").replace("-", "_")
    valid = {"scale_up", "scale_down", "optimize_energy", "migrate_region"}
    # Direct match
    if text in valid:
        return text
    # Search for action in text
    for action in valid:
        if action in text:
            return action
    # Also try partial matches for common model outputs
    partial_map = {
        "scale": "scale_up",
        "up": "scale_up",
        "down": "scale_down",
        "optim": "optimize_energy",
        "energy": "optimize_energy",
        "migrat": "migrate_region",
        "region": "migrate_region",
    }
    for key, action in partial_map.items():
        if key in text:
            return action
    return "optimize_energy"  # safe fallback instead of penalty


def reward_func(completions, **kwargs) -> list[float]:
    """Evaluate each completion by running it through the environment."""
    rewards = []
    for completion in completions:
        # Extract the text from the completion
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        # Run a short episode using the model's suggested actions
        env = EcoCloudEnvironment(difficulty="hard")
        env.reset(seed=42)

        # Parse multiple actions from the text (one per line or comma-separated)
        lines = text.replace(",", "\n").split("\n")
        total_reward = 0.0
        steps = 0

        for line in lines:
            action = extract_action(line)
            if not line.strip():
                continue
            obs = env.step(CloudAction(action=action))
            total_reward += float(obs.last_reward)
            steps += 1
            if obs.done or obs.success:
                break

        # Bonus for taking multiple valid steps
        if steps > 0:
            total_reward += steps * 2.0

        rewards.append(total_reward)

    return rewards


def build_dataset() -> Dataset:
    """Create task prompts for GRPO training."""
    prompts = []
    seeds = [1, 7, 13, 21, 42, 55, 77, 99]

    for i in range(TRAIN_PROMPTS):
        seed = seeds[i % len(seeds)]
        env = EcoCloudEnvironment(difficulty="hard")
        obs = env.reset(seed=seed)

        user_msg = (
            f"Current cloud state:\n"
            f"- latency: {obs.latency:.1f}ms (target: <150)\n"
            f"- cost: ${obs.cost:.1f}/hr (target: <$400)\n"
            f"- carbon: {obs.carbon:.1f} units (target: <220)\n"
            f"- load: {obs.load}\n"
            f"- step: {obs.step_count}/30\n\n"
            f"What sequence of actions should be taken? "
            f"List actions one per line."
        )

        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])

    return Dataset.from_dict({"prompt": prompts})


def main() -> None:
    """Launch GRPO training against the EcoCloud environment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  EcoCloud War Room — GRPO Training")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Prompts: {TRAIN_PROMPTS}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load tokenizer and model explicitly to avoid chat template issues
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=build_dataset(),
        reward_funcs=reward_func,
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_completion_length=256,
            learning_rate=1e-5,
            logging_steps=1,
            save_steps=50,
            num_train_epochs=1,
            log_completions=True,
        ),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    # Save training metadata
    meta = {
        "model": MODEL_NAME,
        "prompts": TRAIN_PROMPTS,
        "completed": datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("  Training complete!")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
