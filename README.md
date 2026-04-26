---
title: CloudEdge
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# ⚡ CloudEdge: Multi-Agent LLM Simulator for Cloud Crises

**Built for the Meta PyTorch OpenEnv Hackathon Grand Finale.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/huggingface/openenv)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**🚀 Live Environment (Hugging Face Space):** [kartikraut09/cloudedge](https://huggingface.co/spaces/kartikraut09/cloudedge)  
**📝 Read the Blog Post:** [Blog.md](Blog.md)

---

## 1. 🎯 The Problem (Domain)
Managing modern cloud infrastructure during a traffic crisis is an impossible balancing act:
- **Want lower latency?** Add servers → *Cost and carbon emissions spike.*
- **Want lower cost?** Remove servers → *Latency degrades, users leave.*
- **Want lower carbon?** Migrate to green regions → *Latency and cost both suffer.*

No single metric can be optimized in isolation. SREs and Cloud Architects struggle with this daily. **CloudEdge** trains an LLM to navigate these competing constraints by negotiating between three specialist AI agents.

---

## 2. 🏗️ The Environment
Built natively on the **OpenEnv** specification, CloudEdge is a 30-step stateful simulation featuring randomized crisis spikes. 

**What the Agent Sees (Observation):**
The agent observes the current `latency` (ms), `cost` ($/hr), `carbon` (emissions), `load` (steady/critical), and the advice of three internal boardroom agents (Resource Agent, Cost Agent, Sustainability Agent). 

**What the Agent Does (Action):**
The agent chooses one of 4 actions: `scale_up`, `scale_down`, `optimize_energy`, or `migrate_region`.

**How the Agent Gets Rewarded:**
We designed a **Shaped Multi-Objective Reward** to prevent reward hacking:
- **Survival Penalty:** The agent accumulates negative rewards (-8, -6, -4) for every step metrics stay above target thresholds (Latency < 150ms, Cost < $400, Carbon < 220). This forces urgency.
- **Gap Closure Bonus:** The agent earns positive rewards (+0.1 per unit) for shrinking the gap between current failing metrics and the target thresholds.

---

## 3. 📊 Results: What Changed After Training?
We fine-tuned **Qwen2.5-0.5B-Instruct** using TRL's **Group Relative Policy Optimization (GRPO)**. Instead of traditional PPO, GRPO evaluated 4 generations per prompt to find the optimal policy without needing a separate value model.

**After 512 training steps on a single Colab T4 GPU:**
- **Reward Skyrocketed:** Average shaped reward improved from **4.6** (untrained random baseline) to **6.8 (+48%)**.
- **Entropy Collapsed:** Action entropy dropped from 0.50 to 0.02, proving the model became highly confident in its policy.
- **Behavior Shift:** The untrained model spammed random actions. The trained model discovered that `optimize_energy` is the mathematical "golden path" to pull down cost and carbon without sacrificing latency, switching to aggressive scaling only when a randomized crisis hit.

![GRPO Training Evidence](training/trl_training_evidence.png)

---

## 4. 🌍 Why Does It Matter?
**Who cares?** Site Reliability Engineers (SREs), Cloud Platform Architects, and Climate/Sustainability Researchers.

**Why?** The tech industry is facing massive scrutiny over the carbon footprint and skyrocketing costs of AI data centers. Current autoscalers only look at CPU/RAM. By proving that an LLM can be trained via RL to successfully manage complex, multi-objective infrastructure crises, CloudEdge paves the way for **autonomous, climate-aware data centers** that heal themselves faster, cheaper, and greener than human operators can.

---

## 🔬 Deep Dive: Technical Implementation & Training Evidence


### Training Pipeline Setup (GRPO)
| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Algorithm | GRPO (Group Relative Policy Optimization) |
| Training Steps | 512 |
| Generations per Prompt | 4 |
| Learning Rate | 5e-6 |
| Temperature | 1.0 |
| Hardware | Google Colab T4 GPU |

### Environment Baseline vs. Trained Comparison
Below are the local simulation graphs demonstrating the agent's ability to maintain stable state across multi-step episodes compared to a random baseline.

**Baseline vs. Trained Performance:**
![Baseline vs Trained](ecocloud_env/graphs/graph2_baseline_vs_trained_20260425_194631.png)

**Crisis Recovery Tracking:**
![Recovery Tracking](ecocloud_env/graphs/graph4_recovery_tracking_20260425_194631.png)

### Anti-Reward-Hacking Measures

| Measure | How It Works |
|---------|-------------|
| **Randomised crisis timing** | Crisis intervals vary per episode — agents can't learn to pre-position for fixed schedules |
| **Repeated action penalty** | -4 reward for repeating the same action consecutively — prevents single-action spam |
| **Anti-oscillation detection** | Boardroom detects scale_up↔scale_down cycles and forces a different action |
| **Multi-objective reward** | No single metric can be gamed — all three must be below target simultaneously |

### Reward Scoring Example
Given a crisis state: `latency=280ms, cost=$620/hr, carbon=380` (all above target):

| Action | Latency Effect | Cost Effect | Carbon Effect | Worst Metric Bonus | **Total Reward** |
|--------|---------------|-------------|---------------|-------------------|-----------------|
| `optimize_energy` | +10 → -0.5 | -20 → +2.0 | -40 → +4.0 | ✅ +2.0 | **+7.5** |
| `scale_down` | +25 → -1.25 | -35 → +3.5 | -15 → +1.5 | ✅ +2.0 | **+5.75** |
| `migrate_region` | +15 → -0.75 | +10 → -0.5 | -50 → +5.0 | ❌ 0 | **+3.75** |
| `scale_up` | -40 → +4.0 | +30 → -1.5 | +20 → -1.0 | ❌ 0 | **+1.5** |

---

## 📁 Project Structure

```
Blog.md                  # Detailed project write-up and training analysis
Dockerfile               # HuggingFace Space deployment configuration
openenv.yaml             # OpenEnv environment manifest
ecocloud_env/
  models.py              # Pydantic v2 state, action, observation models
  agents.py              # ResourceAgent, CostAgent, SustainabilityAgent, Boardroom
  learner.py             # Q-learning controller + adaptive policy wrapper
  training.py            # Training loop with curriculum learning
  training_report.py     # Graph generation (5 publication-quality charts)
  visualize.py           # Entrypoint for training + graphs
  client.py              # OpenEnv client wrapper
  server/
    environment.py       # Core simulation engine (transitions, crises, rewards)
    app.py               # FastAPI / OpenEnv server
dashboard/
  index.html             # Visual real-time dashboard
  style.css              # Premium dark theme
  simulation.js          # JS port of the environment + agents
training/
  trl_grpo_colab.py      # HuggingFace TRL GRPO training script
  requirements-colab.txt # Colab-specific dependencies
notebooks/
  EcoCloud_TRL_GRPO_Colab.ipynb  # Colab notebook entrypoint
run_local.py             # One-episode demo runner
requirements.txt         # Project dependencies
```

---

## 🏃 Quick Start & Links

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the OpenEnv API server & Visual Dashboard
uvicorn ecocloud_env.server.app:app --host 0.0.0.0 --port 7860
# (Then open http://localhost:7860 in your browser)

# 3. Run a local simulation episode (Auto-detects trained policy)
python run_local.py

# 4. Compare untrained vs trained policies
python run_local.py heuristic   # Runs the baseline
python run_local.py trained     # Runs the trained Q-policy

# 5. Re-run local training & generate new graphs
python ecocloud_env/visualize.py
```

- **Hugging Face Space:** [kartikraut09/cloudedge](https://huggingface.co/spaces/kartikraut09/cloudedge)
- **Trained Model:** [kartikraut09/ecocloud-grpo-qwen](https://huggingface.co/kartikraut09/ecocloud-grpo-qwen)
- **Training Notebook:** `notebooks/EcoCloud_TRL_GRPO_Colab.ipynb`
