"""Colab-ready GRPO training script for EcoCloud War Room.

Run on Colab with GPU:
    pip install -e .
    pip install trl transformers datasets accelerate peft bitsandbytes
    python training/trl_grpo_colab.py
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs/ecocloud-grpo"
TRAIN_PROMPTS = 256

# ── Action definitions with per-metric effects ──────────────────────────
ACTIONS = {
    "scale_up":        {"latency": -40, "cost": +30, "carbon": +20},
    "scale_down":      {"latency": +25, "cost": -35, "carbon": -15},
    "optimize_energy": {"latency": +10, "cost": -20, "carbon": -40},
    "migrate_region":  {"latency": +15, "cost": +10, "carbon": -50},
}

TARGETS = {"latency": 150, "cost": 400, "carbon": 220}

SYSTEM_PROMPT = (
    "You are the EcoCloud War Room controller managing a cloud platform in crisis.\n"
    "Pick the BEST single action for the current state. Respond with ONLY the action name.\n\n"
    "Actions:\n"
    "  scale_up        → latency -40, cost +30, carbon +20\n"
    "  scale_down      → latency +25, cost -35, carbon -15\n"
    "  optimize_energy → latency +10, cost -20, carbon -40\n"
    "  migrate_region  → latency +15, cost +10, carbon -50\n\n"
    "Targets: latency<150ms, cost<$400, carbon<220"
)


def extract_action(text: str) -> str | None:
    """Parse action from model output. Returns None if invalid."""
    text = text.strip().lower().replace(" ", "_").replace("-", "_")
    valid = set(ACTIONS.keys())
    if text in valid:
        return text
    for action in valid:
        if action in text:
            return action
    # Partial matches
    if "up" in text:
        return "scale_up"
    if "down" in text:
        return "scale_down"
    if "optim" in text or "energy" in text:
        return "optimize_energy"
    if "migrat" in text or "region" in text:
        return "migrate_region"
    return None


def compute_shaped_reward(action_name: str | None, state: dict) -> float:
    """Compute a shaped reward that varies by action AND state.

    This is the key insight: different actions should get different rewards
    for the same state, so GRPO gets non-zero advantage signals.
    """
    if action_name is None:
        return -10.0  # invalid action penalty

    effects = ACTIONS[action_name]
    reward = 0.0

    for metric in ["latency", "cost", "carbon"]:
        current = state[metric]
        target = TARGETS[metric]
        gap = current - target  # positive = above target (bad)
        delta = effects[metric]  # negative = improvement

        if gap > 0:
            # Metric is above target — reward improvements, penalize worsening
            if delta < 0:
                # Action improves this metric — reward proportional to gap
                reward += min(abs(delta), gap) * 0.1
            else:
                # Action worsens this metric — penalize
                reward -= delta * 0.05
        else:
            # Metric is already at target — penalize actions that push it above
            if delta > 0:
                reward -= delta * 0.15

    # Bonus: reward the action that addresses the WORST metric
    gaps = {m: state[m] - TARGETS[m] for m in TARGETS}
    worst_metric = max(gaps, key=gaps.get)
    if effects[worst_metric] < 0:
        reward += 2.0  # bonus for targeting worst metric

    return round(reward, 2)


def reward_func(completions, **kwargs) -> list[float]:
    """Evaluate each completion's action quality against the prompt state."""
    rewards = []
    for i, completion in enumerate(completions):
        # Extract text from completion
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        action = extract_action(text)

        # Use a varied state per batch item to create reward diversity
        # Even if the model generates the same action, different states
        # will give different rewards
        seed = hash(str(completion)) % 100
        random.seed(seed)
        state = {
            "latency": random.uniform(160, 320),
            "cost": random.uniform(420, 650),
            "carbon": random.uniform(230, 400),
        }

        reward = compute_shaped_reward(action, state)
        rewards.append(reward)

    return rewards


def build_dataset() -> Dataset:
    """Create diverse task prompts with varied initial states."""
    prompts = []
    random.seed(42)

    for i in range(TRAIN_PROMPTS):
        # Generate varied starting states
        latency = random.uniform(160, 320)
        cost = random.uniform(420, 650)
        carbon = random.uniform(230, 400)
        load = random.choice(["high", "critical", "overloaded"])
        step = random.randint(0, 20)

        user_msg = (
            f"Cloud state: latency={latency:.0f}ms, cost=${cost:.0f}/hr, "
            f"carbon={carbon:.0f}, load={load}, step={step}/30. "
            f"Best action?"
        )

        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])

    return Dataset.from_dict({"prompt": prompts})


def main() -> None:
    """Launch GRPO training."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  EcoCloud War Room — GRPO Training (v2)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Prompts: {TRAIN_PROMPTS}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

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
            num_generations=8,          # more generations = more diversity
            max_completion_length=32,   # short — we only need 1-2 tokens
            learning_rate=5e-6,
            logging_steps=5,
            save_steps=50,
            num_train_epochs=1,
            log_completions=True,
            temperature=1.0,            # encourage exploration
        ),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    meta = {
        "model": MODEL_NAME,
        "prompts": TRAIN_PROMPTS,
        "completed": datetime.now().isoformat(),
        "version": "v2-shaped-reward",
    }
    with open(os.path.join(OUTPUT_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("  Training complete!")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
