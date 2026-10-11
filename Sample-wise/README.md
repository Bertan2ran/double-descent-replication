# Experiment 2: Sample-Wise Non-Monotonicity

Replication of Figure 12 (Right) from Nakkiran et al. evaluating sample-wise double descent across $n \in [6250, 50000]$ with a 5-layer CNN ($k=24$) under 20% label noise.

## 1. Quick Reproduction (Plotting Existing Results)
The raw training logs across all 10 sample configurations are pre-computed in `results/`. To recreate Figure 2 from the report:

```bash
python plot_sample_descent.py
