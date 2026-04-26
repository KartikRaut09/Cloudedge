---
license: mit
language:
- en
tags:
- reinforcement-learning
- grpo
- cloud-management
- multi-agent
- sustainability
- openenv
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: text-generation
---

# ⚡ CloudEdge GRPO Controller

**A Qwen2.5-0.5B model fine-tuned with Group Relative Policy Optimization (GRPO) to manage cloud infrastructure crises.**

Built for the **Meta PyTorch OpenEnv Hackathon Grand Finale**.

## Model Description

This model is a reinforcement-learning-trained controller for the **CloudEdge** cloud crisis simulator. It learns to select optimal infrastructure actions (scale_up, scale_down, optimize_energy, migrate_region) by balancing three competing objectives:

| Objective | Target | Agent |
|-----------|--------|-------|
| Latency | < 150ms | ResourceAgent |
| Cost | < $400/hr | CostAgent |
| Carbon | < 220 units | SustainabilityAgent |

### Training Method

- **Algorithm:** GRPO (Group Relative Policy Optimization) via [TRL](https://github.com/huggingface/trl)
- **Base Model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Training Steps:** 512
- **Generations per prompt:** 4
- **Reward Function:** Shaped multi-objective reward with gap closure + worst-metric bonus

### Shaped Reward Function

```
reward = Σ (gap_closure × weight) + worst_metric_bonus
```

| Action | Reward (crisis state) | Model Learns |
|--------|----------------------|-------------|
| `optimize_energy` | **+7.5** | "Best action — addresses cost + carbon simultaneously" |
| `scale_down` | **+5.75** | "Good — reduces cost effectively" |
| `migrate_region` | **+3.75** | "Moderate — helps carbon but hurts cost" |
| `scale_up` | **+1.5** | "Worst — increases cost and carbon" |

### Training Results

The model converged to selecting `optimize_energy` as the dominant policy when all metrics are above target — which is the mathematically optimal action given the shaped reward function.

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("kartikraut09/ecocloud-grpo-qwen")
tokenizer = AutoTokenizer.from_pretrained("kartikraut09/ecocloud-grpo-qwen")

prompt = """<|im_start|>system
You are the CloudEdge controller managing a cloud platform in crisis.
Pick the BEST single action for the current state. Respond with ONLY the action name.

Actions:
  scale_up        → latency -40, cost +30, carbon +20
  scale_down      → latency +25, cost -35, carbon -15
  optimize_energy → latency +10, cost -20, carbon -40
  migrate_region  → latency +15, cost +10, carbon -50

Targets: latency<150ms, cost<$400, carbon<220<|im_end|>
<|im_start|>user
Cloud state: latency=280ms, cost=$620/hr, carbon=380, load=critical. Best action?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=16, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: optimize_energy
```

## Technical Details

- **Architecture:** Qwen2 (0.5B parameters)
- **Framework:** PyTorch + HuggingFace Transformers + TRL
- **Environment:** OpenEnv-compatible Gymnasium-style simulator
- **Training Hardware:** Google Colab T4 GPU
- **Training Time:** ~15 minutes (512 steps)

## Project Links

- **GitHub:** [KartikRaut09/ecocloud-war-room](https://github.com/KartikRaut09/ecocloud-war-room)
- **Hackathon:** Meta PyTorch OpenEnv Hackathon Grand Finale
- **Themes:** Multi-Agent Interactions · Long-Horizon Planning · World Modeling

## Citation

```bibtex
@misc{cloudedge2026,
  title={CloudEdge: Multi-Agent LLM Simulator for Sustainable Cloud Crisis Management},
  author={Kartik Raut},
  year={2026},
  url={https://github.com/KartikRaut09/ecocloud-war-room}
}
```

## License

MIT License
