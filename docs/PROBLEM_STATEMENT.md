# Problem Statement and MDP Formulation

## Objective

TA-RWARE models a warehouse in which automated guided vehicles (AGVs) collect items from racks and deliver them to goal docks while picker agents assist near delivery areas. The goal is to learn a dispatch and navigation policy that maximizes completed deliveries while minimizing collisions, wasted movement, battery failures, and idle time.

## Novelty

TA-RWARE extends the standard RWARE-style warehouse problem with:

1. Battery dynamics and autonomous charging behavior.
2. Heterogeneous agents: AGVs perform deliveries, pickers provide assist rewards near docks.
3. Auto-pickup and auto-drop transitions that remove the sparse `ACT` bottleneck.
4. Dynamic order injection during an episode.
5. Interactive Streamlit evaluation with DQN-vs-baseline comparison.

These choices make the environment closer to a small fulfillment warehouse where energy limits, dock congestion, and changing orders matter.

## MDP Definition

The task is modeled as a finite-horizon Markov Decision Process:

```text
MDP = (S, A, T, R, gamma)
```

| Element | Definition in TA-RWARE |
|---|---|
| State `S` | Per-agent observation vector containing own state, target direction, order context, nearby agents, and local grid view |
| Action `A` | Discrete movement action: north, south, east, west, wait |
| Transition `T` | Warehouse physics, collision checks, battery drain/recharge, auto-pickup/drop, order assignment, task injection |
| Reward `R` | Delivery, pickup, progress shaping, collision penalty, time penalty, battery penalty, picker assist, team bonus |
| Discount `gamma` | `0.97`, favoring long-term delivery completion while still valuing near-term movement choices |

## State Space

For the main configuration, each agent receives a 57-dimensional normalized observation.

| Dimensions | Content | Range |
|---:|---|---|
| 0-4 | Own x/y position, battery ratio, carrying flag, phase | mostly `[0, 1]` |
| 5-7 | Target delta x, target delta y, Manhattan distance | `[-1, 1]` |
| 8-11 | Assigned rack and goal coordinates | `[0, 1]` |
| 12-19 | Up to four pending order rack positions | `[0, 1]` |
| 20-31 | Up to four nearest agents: delta x, delta y, carrying flag | `[-1, 1]` |
| 32-56 | Local 5x5 grid perception | `[0, 1]` |

This representation is compact enough for DQN while still preserving the information needed for one-step routing decisions.

## Action Space

```text
0 = North
1 = South
2 = East
3 = West
4 = Wait
```

Pickup and drop are automatic when an AGV reaches its assigned rack or goal. This keeps the learning task focused on navigation and reduces delayed credit assignment.

## Transition Function

The environment transitions are deterministic given the current joint state and joint action, except for randomized order generation during reset and task injection.

- Valid movement: agent moves one cell and battery decreases by `battery_drain_move`.
- Invalid movement: agent stays in place, receives collision penalty, and battery decreases by `battery_drain_idle`.
- Rack entry: only an AGV assigned to that rack may enter it.
- Auto-pickup: when an AGV reaches its rack during `TO_PICKUP`, phase changes to `TO_GOAL`.
- Auto-drop: when an AGV reaches its goal during `TO_GOAL`, the order is completed.
- Low battery: if battery ratio falls below threshold, the agent routes toward a charging cell.
- Charging: battery increases by `battery_recharge_rate` until full.
- Task injection: new orders are added every `task_injection_interval` steps.
- Episode ends when all orders are completed or `max_steps` is reached.

## Reward Function

| Reward Component | Value | Purpose |
|---|---:|---|
| Delivery complete | `+32.0` | Main terminal success signal |
| Pickup item | `+8.0` | Intermediate milestone for reaching rack |
| Progress | `+0.6` | Dense guidance for moving closer to target |
| Move away | `-1.2` | Stronger penalty to reduce oscillation |
| Collision | `-1.0` | Discourages invalid moves and congestion |
| Time step | `-0.012` | Soft pressure for faster completion |
| Battery empty | `-6.0` | Penalizes energy failure |
| Picker assist | `+2.5` | Rewards pickers near successful deliveries |
| Team bonus | `+55.0` | Cooperative completion signal |

## Scale Justification

The default 5x7 rack layout with 3 AGVs and 2 pickers represents a small-to-medium warehouse zone. The layout is large enough to create routing conflicts, battery decisions, and multi-agent coordination, but small enough for CPU-based DQN training and live Streamlit demonstration.
