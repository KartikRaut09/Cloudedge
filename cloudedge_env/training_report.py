"""Generate true training graphs and summaries for CloudEdge."""

from __future__ import annotations

import os
import sys
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    from learner import learner_payload
    from training import evaluate_policy, train_policy
except ImportError:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from ecocloud_env.learner import learner_payload
    from ecocloud_env.training import evaluate_policy, train_policy

GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "graphs")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
POLICY_PATH = os.path.join(ARTIFACTS_DIR, "boardroom_q_policy.json")


def save_fig(fig: plt.Figure, filename_prefix: str) -> str:
    """Save figure as PNG with timestamp label in filename."""
    filename = f"{filename_prefix}_{timestamp}.png"
    filepath = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {filepath}")
    return filepath


def rolling_avg(data: list[float], window: int = 10) -> list[float]:
    """Compute rolling average with given window size."""
    return [np.mean(data[max(0, i - window + 1) : i + 1]) for i in range(len(data))]


def graph1(training: dict[str, list[float] | list[int]]) -> None:
    """Save the main four-panel training curve."""
    rewards = training["eval_rewards"]
    success = training["eval_success"]
    latency = training["eval_final_latency"]
    cost = training["eval_final_cost"]
    x = np.arange(1, len(rewards) + 1)
    lat_roll = np.array(rolling_avg(latency))
    cost_roll = np.array(rolling_avg(cost))
    success_roll = [np.mean(success[max(0, i - 9) : i + 1]) * 100 for i in range(len(success))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "CloudEdge - Training Progress\n"
        f"Q-learning over 60 Episodes | Generated: {timestamp}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    axes[0][0].fill_between(x, rewards, alpha=0.2, color="#7F77DD")
    axes[0][0].plot(x, rewards, color="#7F77DD", linewidth=1, alpha=0.65, label="Per episode")
    axes[0][0].plot(x, rolling_avg(rewards), color="#3C3489", linewidth=2.5, label="10-ep rolling avg")
    axes[0][0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0][0].axhline(30, color="green", linestyle="--", linewidth=1.5, label="Target reward")
    axes[0][0].set(title="Greedy Eval Reward per Episode", xlabel="Episode", ylabel="Reward")
    axes[0][0].legend()
    axes[0][0].grid(alpha=0.3)
    axes[0][1].bar(x, success_roll, color="#7BC67B", alpha=0.6)
    axes[0][1].plot(x, success_roll, color="#1F6F43", linewidth=2, label="Rolling success")
    axes[0][1].axhline(50, color="red", linestyle="--", label="50% target")
    axes[0][1].axhline(80, color="orange", linestyle="--", label="80% stretch")
    axes[0][1].set(title="Success Rate (rolling 10-episode %)", xlabel="Episode", ylabel="Success %", ylim=(0, 115))
    axes[0][1].legend()
    axes[0][1].grid(alpha=0.3)
    axes[1][0].plot(x, latency, color="#F2A7A0", alpha=0.5, label="Per episode")
    axes[1][0].plot(x, lat_roll, color="#A62C2C", linewidth=2.5, label="10-ep rolling avg")
    axes[1][0].axhline(150, color="red", linestyle="--", linewidth=2, label="Target: 150ms")
    axes[1][0].fill_between(x, lat_roll, 150, where=lat_roll > 150, color="red", alpha=0.1)
    axes[1][0].fill_between(x, lat_roll, 150, where=lat_roll <= 150, color="green", alpha=0.1)
    axes[1][0].set(title="Greedy Eval Final Latency", xlabel="Episode", ylabel="Latency (ms)")
    axes[1][0].legend()
    axes[1][0].grid(alpha=0.3)
    axes[1][1].plot(x, cost, color="#F6C667", alpha=0.5, label="Per episode")
    axes[1][1].plot(x, cost_roll, color="#C76B00", linewidth=2.5, label="10-ep rolling avg")
    axes[1][1].axhline(400, color="red", linestyle="--", linewidth=2, label="Target: $400")
    axes[1][1].fill_between(x, cost_roll, 400, where=cost_roll > 400, color="red", alpha=0.1)
    axes[1][1].fill_between(x, cost_roll, 400, where=cost_roll <= 400, color="green", alpha=0.1)
    axes[1][1].set(title="Greedy Eval Final Cost", xlabel="Episode", ylabel="Cost ($)")
    axes[1][1].legend()
    axes[1][1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "graph1_training_curve")
    plt.close(fig)


def graph2(
    baseline: dict[str, list[float] | list[int]],
    trained: dict[str, list[float] | list[int]],
) -> None:
    """Save baseline-vs-trained comparison bars."""
    baseline_reward = np.mean(baseline["episode_rewards"])
    trained_reward = np.mean(trained["episode_rewards"])
    baseline_success = np.mean(baseline["episode_success"]) * 100
    trained_success = np.mean(trained["episode_success"]) * 100
    baseline_latency = np.mean(baseline["episode_final_latency"])
    trained_latency = np.mean(trained["episode_final_latency"])
    baseline_cost = np.mean(baseline["episode_final_cost"])
    trained_cost = np.mean(trained["episode_final_cost"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "CloudEdge - Baseline vs Trained Policy\n"
        "Heuristic Boardroom vs Q-Learned Boardroom",
        fontsize=13,
        fontweight="bold",
    )
    bars = axes[0].bar(["Baseline", "Trained"], [baseline_reward, trained_reward], color=["#F09595", "#5DCAA5"], width=0.5, edgecolor="none")
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}", ha="center", va="bottom")
    axes[0].annotate(f"{trained_reward - baseline_reward:+.1f} reward", xy=(1, trained_reward), xytext=(0.45, max(baseline_reward, trained_reward) + 8), arrowprops={"arrowstyle": "->", "color": "#3C3489"}, color="#3C3489")
    axes[0].set(title="Average Reward", ylabel="Reward")
    axes[0].grid(axis="y", alpha=0.3)
    bars = axes[1].bar(["Baseline", "Trained"], [baseline_success, trained_success], color=["#F09595", "#5DCAA5"], width=0.5, edgecolor="none")
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.0f}%", ha="center", va="bottom")
    axes[1].axhline(50, color="red", linestyle="--", linewidth=1.5)
    axes[1].set(title="Success Rate", ylabel="Success %", ylim=(0, 110))
    axes[1].grid(axis="y", alpha=0.3)
    width = 0.3
    axes[2].bar([0 - width / 2, 1 - width / 2], [baseline_latency / 10, baseline_cost / 10], width=width, label="Baseline", color="#F09595", edgecolor="none")
    axes[2].bar([0 + width / 2, 1 + width / 2], [trained_latency / 10, trained_cost / 10], width=width, label="Trained", color="#5DCAA5", edgecolor="none")
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Latency (ms)", "Cost ($/10)"])
    axes[2].axhline(15, color="red", linestyle="--", linewidth=1.5)
    axes[2].axhline(40, color="orange", linestyle="--", linewidth=1.5)
    axes[2].text(-0.18, baseline_latency / 10, f"{baseline_latency:.0f}")
    axes[2].text(0.08, trained_latency / 10, f"{trained_latency:.0f}")
    axes[2].text(0.82, baseline_cost / 10, f"${baseline_cost:.0f}")
    axes[2].text(1.08, trained_cost / 10, f"${trained_cost:.0f}")
    axes[2].set(title="Latency & Cost Comparison")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "graph2_baseline_vs_trained")
    plt.close(fig)


def graph3(training: dict[str, list[float] | list[int]]) -> None:
    """Save a step-by-step comparison of the first and final training episodes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "CloudEdge - Episode 1 vs Final Training Episode\n"
        "Shows how the learned policy changes behaviour",
        fontsize=13,
        fontweight="bold",
    )
    steps1 = range(1, len(training["ep1_step_rewards"]) + 1)
    steps_last = range(1, len(training["eplast_step_rewards"]) + 1)
    axes[0].plot(steps1, training["ep1_step_rewards"], color="#F07F6A", label="Episode 1")
    axes[0].plot(steps_last, training["eplast_step_rewards"], color="#1F9E89", label="Episode 60")
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set(title="Step Reward", xlabel="Step", ylabel="Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps1, training["ep1_latencies"], color="#F07F6A", label="Episode 1")
    axes[1].plot(steps_last, training["eplast_latencies"], color="#1F9E89", label="Episode 60")
    axes[1].axhline(150, color="red", linestyle="--", label="Target 150ms")
    axes[1].set(title="Latency per Step", xlabel="Step", ylabel="Latency (ms)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(steps1, training["ep1_costs"], color="#F07F6A", label="Episode 1")
    axes[2].plot(steps_last, training["eplast_costs"], color="#1F9E89", label="Episode 60")
    axes[2].axhline(400, color="red", linestyle="--", label="Target $400")
    axes[2].set(title="Cost per Step", xlabel="Step", ylabel="Cost ($)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "graph3_episode_comparison")
    plt.close(fig)


def graph4(training: dict[str, list[float] | list[int]]) -> None:
    """Save carbon, stability, migrate usage, and crisis usage plots."""
    carbon = training["eval_final_carbon"]
    stable = training["episode_stable_steps"]
    migrate = training["migrate_count_per_episode"]
    crisis = training["crisis_response_count_per_episode"]
    x = np.arange(1, len(carbon) + 1)
    carbon_roll = np.array(rolling_avg(carbon))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "CloudEdge - Sustainability and Recovery Signals During Training",
        fontsize=13,
        fontweight="bold",
    )
    axes[0][0].plot(x, carbon, color="#A9D18E", alpha=0.5, label="Per episode")
    axes[0][0].plot(x, carbon_roll, color="#2E7D32", linewidth=2.5, label="10-ep rolling avg")
    axes[0][0].axhline(220, color="red", linestyle="--", label="Success target: 220")
    axes[0][0].fill_between(x, carbon_roll, 220, where=carbon_roll > 220, color="red", alpha=0.1)
    axes[0][0].fill_between(x, carbon_roll, 220, where=carbon_roll <= 220, color="green", alpha=0.1)
    axes[0][0].legend(handles=[axes[0][0].lines[0], axes[0][0].lines[1], axes[0][0].lines[2], mpatches.Patch(color="green", alpha=0.1, label="Below target")])
    axes[0][0].set(title="Final Carbon per Episode", xlabel="Episode", ylabel="Carbon (units)")
    axes[0][0].grid(alpha=0.3)
    axes[0][1].bar(x, stable, color="#7F77DD", alpha=0.7, label="Per episode")
    axes[0][1].plot(x, rolling_avg(stable), color="#3C3489", linewidth=2.5, label="10-ep rolling avg")
    axes[0][1].axhline(5, color="orange", linestyle="--", label="Stable bonus threshold")
    axes[0][1].set(title="Consecutive Stable Steps", xlabel="Episode", ylabel="Stable Steps")
    axes[0][1].legend()
    axes[0][1].grid(alpha=0.3)
    axes[1][0].bar(x, migrate, color="#1F9E89", alpha=0.75, label="migrate_region uses")
    axes[1][0].plot(x, rolling_avg(migrate), color="#0B5D56", linewidth=2.5, label="10-ep rolling avg")
    axes[1][0].set(title="Migrate Region Usage", xlabel="Episode", ylabel="Action Count")
    axes[1][0].legend()
    axes[1][0].grid(alpha=0.3)
    axes[1][1].bar(x, crisis, color="#4B6CB7", alpha=0.75, label="crisis_response uses")
    axes[1][1].plot(x, rolling_avg(crisis), color="#233A7A", linewidth=2.5, label="10-ep rolling avg")
    axes[1][1].set(title="Crisis Response Usage", xlabel="Episode", ylabel="Action Count")
    axes[1][1].legend()
    axes[1][1].grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "graph4_recovery_tracking")
    plt.close(fig)


def graph5(
    training: dict[str, list[float] | list[int]],
    baseline: dict[str, list[float] | list[int]],
    trained: dict[str, list[float] | list[int]],
) -> None:
    """Save the summary table image for README and submission use."""
    baseline_reward = np.mean(baseline["episode_rewards"])
    trained_reward = np.mean(trained["episode_rewards"])
    baseline_success = np.mean(baseline["episode_success"]) * 100
    trained_success = np.mean(trained["episode_success"]) * 100
    baseline_latency = np.mean(baseline["episode_final_latency"])
    trained_latency = np.mean(trained["episode_final_latency"])
    baseline_cost = np.mean(baseline["episode_final_cost"])
    trained_cost = np.mean(trained["episode_final_cost"])
    baseline_carbon = np.mean(baseline["episode_final_carbon"])
    trained_carbon = np.mean(trained["episode_final_carbon"])
    best_episode = int(training["best_checkpoint"][0])
    best_eval_reward = float(training["best_eval_reward"][0])
    table_data = [
        ["Metric", "Baseline", "Trained", "Change"],
        ["Avg Reward", f"{baseline_reward:.1f}", f"{trained_reward:.1f}", f"{trained_reward - baseline_reward:+.1f}"],
        ["Success Rate", f"{baseline_success:.0f}%", f"{trained_success:.0f}%", f"{trained_success - baseline_success:+.0f}pp"],
        ["Avg Final Latency", f"{baseline_latency:.0f} ms", f"{trained_latency:.0f} ms", f"{trained_latency - baseline_latency:+.0f} ms"],
        ["Avg Final Cost", f"${baseline_cost:.0f}", f"${trained_cost:.0f}", f"{trained_cost - baseline_cost:+.0f}"],
        ["Avg Final Carbon", f"{baseline_carbon:.0f}", f"{trained_carbon:.0f}", f"{trained_carbon - baseline_carbon:+.0f}"],
        ["Best Eval Checkpoint", "Episodes 1-10", f"Ep {best_episode}", f"{best_eval_reward:+.1f}"],
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    fig.suptitle("CloudEdge - Training Summary Report", fontsize=13, fontweight="bold", y=0.98)
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for col in range(4):
        table[0, col].set_facecolor("#3C3489")
        table[0, col].set_text_props(color="white", fontweight="bold")
    for row in range(1, len(table_data)):
        for col in range(4):
            table[row, col].set_facecolor("#F1EFE8" if row % 2 == 0 else "white")
        metric = table_data[row][0]
        value = table_data[row][3]
        is_good_reduction = value.startswith("-") and metric in {"Avg Final Latency", "Avg Final Cost", "Avg Final Carbon"}
        is_positive_gain = value.startswith("+") and metric not in {"Avg Final Latency", "Avg Final Cost", "Avg Final Carbon"}
        table[row, 3].set_text_props(color="#085041" if (is_good_reduction or is_positive_gain) else "#B33030", fontweight="bold")
    table.auto_set_column_width([0, 1, 2, 3])
    plt.tight_layout()
    save_fig(fig, "graph5_summary_table")
    plt.close(fig)


def main() -> None:
    """Train the learner, evaluate baseline vs trained, and save all graphs."""
    learner, training_metrics = train_policy(episodes=60)
    learner.save(POLICY_PATH)
    baseline_results = evaluate_policy(episodes=20, learner=None)
    trained_results = evaluate_policy(episodes=20, learner=learner)
    graph1(training_metrics)
    graph2(baseline_results, trained_results)
    graph3(training_metrics)
    graph4(training_metrics)
    graph5(training_metrics, baseline_results, trained_results)
    payload = learner_payload(learner)
    best_episode = int(training_metrics["best_checkpoint"][0])
    best_eval_reward = float(training_metrics["best_eval_reward"][0])
    best_eval_success = max(training_metrics["eval_success"]) * 100
    print()
    print("=" * 60)
    print("  CLOUDEDGE - TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Graphs saved to : {GRAPHS_DIR}")
    print(f"  Policy saved to : {POLICY_PATH}")
    print()
    print("  Files generated:")
    print(f"    graph1_training_curve_{timestamp}.png")
    print(f"    graph2_baseline_vs_trained_{timestamp}.png")
    print(f"    graph3_episode_comparison_{timestamp}.png")
    print(f"    graph4_recovery_tracking_{timestamp}.png")
    print(f"    graph5_summary_table_{timestamp}.png")
    print()
    print("  Training stats:")
    print(f"    Episodes trained   : 60")
    print(f"    States learned     : {payload['states_learned']}")
    print(f"    Final epsilon      : {payload['epsilon']:.4f}")
    print(f"    Early eval reward  : {np.mean(training_metrics['eval_rewards'][:10]):.1f}")
    print(f"    Best eval reward   : {best_eval_reward:.1f} (episode {best_episode})")
    print(f"    Early eval success : {np.mean(training_metrics['eval_success'][:10]) * 100:.0f}%")
    print(f"    Best eval success  : {best_eval_success:.0f}%")
    print(f"    Baseline eval succ.: {np.mean(baseline_results['episode_success']) * 100:.0f}%")
    print(f"    Trained eval succ. : {np.mean(trained_results['episode_success']) * 100:.0f}%")
    print("=" * 60)
    print()
    print("  Use these outputs in your:")
    print("    - README.md (embed graph1 and graph2)")
    print("    - Colab notebook report")
    print("    - Demo video and slide deck")
    print("    - Hugging Face Space README")
    print("=" * 60)


if __name__ == "__main__":
    main()
