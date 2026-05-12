# Result Analysis and Interpretation

## Deterministic Policy Comparison

The current saved checkpoints were evaluated using three deterministic seeds (`1234`, `1235`, `1236`). Artifacts are saved in:

- `docs/results/policy_comparison_summary.csv`
- `docs/results/policy_comparison_raw.csv`
- `docs/results/policy_comparison.png`

## Smoke Training Verification

A short reproducibility run was executed with `configs/config.smoke.yaml`:

```powershell
venv\Scripts\python.exe train.py --config configs\config.smoke.yaml --run_name rubric_smoke_verify
```

The run completed successfully with global seed `42`, produced metrics, and generated training plots. Because runtime logs are gitignored, the important verification artifacts were copied to:

- `docs/results/smoke_metrics.json`
- `docs/results/smoke_training_metrics.png`
- `docs/results/smoke_training_metrics_agents.png`

| Rank by reward | Policy | Avg Reward | Avg Deliveries | Completion Rate | Avg Collisions |
|---:|---|---:|---:|---:|---:|
| 1 | DQN | `+154.59` | `5.00` | `83.3%` | `100.00` |
| 2 | Greedy | `-5.47` | `0.33` | `3.3%` | `0.33` |
| 3 | Random | `-1653.67` | `0.00` | `0.0%` | `623.00` |

## Interpretation

The DQN checkpoint ranks highest by reward and completes substantially more orders than both baselines, which shows that it has learned goal-directed delivery behavior. It also has more collisions than the greedy policy, which reveals a realistic safety/coordination tradeoff in the current independent-learner setup.

This is a useful result for viva discussion: the model learned the task objective but needs better safety and coordination. The result supports two conclusions:

1. DQN learned higher-throughput behavior than simple heuristics.
2. Independent learners still need stronger coordination or collision-aware shaping.

## Policy Behavior

- DQN aggressively pursues deliveries and reaches more goals.
- Greedy is conservative and has low collision count, but completes fewer orders.
- Random performs poorly on reward, deliveries, and collisions, confirming that the environment is non-trivial.

## Failure Modes

1. High DQN collision rate indicates local congestion and invalid rack/wall moves.
2. Independent policies do not coordinate dock access explicitly.
3. Reward weighting may overemphasize delivery progress compared to collision avoidance.

## Improvements Suggested by Results

- Increase collision penalty or add path-blocking awareness.
- Add centralized training such as QMIX or MAPPO.
- Add prioritized replay so rare but important transitions are sampled more often.
- Add curriculum training: start with one AGV, then scale to multiple agents.
