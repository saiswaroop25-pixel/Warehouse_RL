# Algorithmic Design and Model Architecture

## Chosen Algorithm

TA-RWARE uses independent Dueling Double DQN agents. Each warehouse agent owns a DQN policy, observes its local state vector, and selects one of five discrete actions.

## Why Dueling Double DQN?

| Algorithm | Strength | Weakness in this project | Decision |
|---|---|---|---|
| Plain DQN | Simple and compatible with discrete actions | Conflates state value and action advantage | Rejected |
| PPO | Stable policy optimization | On-policy and sample hungry for this small CPU project | Rejected |
| QMIX | Strong centralized-training/decentralized-execution MARL method | More complex mixer and credit assignment machinery | Future work |
| Independent Q-learning | Simple multi-agent baseline | Less expressive than dueling architecture | Rejected |
| Dueling Double DQN | Separates value and advantage; reduces overestimation through Double DQN target | Independent learners can still collide | Chosen |

Warehouse routing has many states where several actions are nearly equivalent or equally bad: blocked corridors, charging states, and close-range dock congestion. Dueling DQN helps by learning how good the state is separately from which action is marginally best.

Double DQN reduces overestimation bias by selecting the next action with the policy network while evaluating it with the target network.

## Architecture

The main model uses hidden dimensions:

```text
state_dim -> 256 -> 128 -> value head 64 -> 1
                         -> advantage head 64 -> action_dim
```

The first hidden layer is larger than `2 x state_dim`, which gives enough capacity for local grid features, battery context, and task context. The later layers reduce dimensionality before splitting into value and advantage heads.

## Independent Learners

Independent learners were chosen because the project has heterogeneous roles: AGVs deliver orders while pickers provide assist behavior. A single shared policy would mix these roles unless additional role encodings and parameter-sharing logic were added.

This choice is simple and explainable for a course project, but it also creates a known limitation: agents can still interfere with each other because there is no centralized coordination module.

## Reward Shaping Rationale

The reward function combines sparse terminal rewards with dense shaping:

- `delivery_complete` and `team_bonus` define the final task objective.
- `pickup_item` gives a meaningful intermediate milestone.
- `progress` and `move_away` guide navigation before delivery rewards are observed.
- `collision`, `time_step`, and `battery_empty` discourage unsafe or inefficient behavior.

The move-away penalty is stronger than the progress reward so that backtracking and oscillation are not profitable.

## Baselines

Two baselines are included:

- Random policy: lower-bound behavior.
- Greedy policy: nearest-target heuristic with no learned long-term value.

These baselines make it possible to interpret whether DQN is learning useful behavior beyond simple movement heuristics.

## Future Architecture Improvements

1. QMIX or MAPPO for centralized multi-agent credit assignment.
2. Prioritized replay to sample rare team-bonus and battery-failure events more often.
3. Graph neural network observations for full warehouse connectivity.
4. Parameter sharing with role embeddings for better sample efficiency.
