# RL Concepts Used in TA-RWARE

## Why Reinforcement Learning?

Warehouse routing is sequential. An action that looks good now can block another agent later, waste battery, or delay a delivery. Reinforcement learning is suitable because agents learn from interaction and delayed reward instead of requiring a labeled dataset of optimal moves.

## Bellman Equation

The project uses the Q-learning target:

```text
Q*(s, a) = E[R(s, a, s') + gamma max_a' Q*(s', a')]
```

In this project:

- `s` is the normalized warehouse observation.
- `a` is one of five movement actions.
- `R` is the sum of delivery, pickup, progress, collision, battery, and team rewards.
- `gamma = 0.97`, so future deliveries matter strongly.

## Exploration vs Exploitation

Training starts with `epsilon = 1.0`, meaning actions are mostly random. Epsilon decays slowly so agents explore enough rack, goal, charger, and congestion patterns. It ends at `0.05`, preserving some exploration for dynamic order patterns.

## Credit Assignment

Auto-pickup and auto-drop reduce the difficulty of credit assignment. Without them, the agent must learn both navigation and a separate activation action at exactly the correct cell. Auto transitions make success depend mostly on reaching the right location, giving a cleaner reward signal.

## Multi-Agent Learning

Each agent learns independently, but the environment is shared. This means one agent's action changes the next state observed by others. The team bonus encourages cooperation, while collision penalties discourage harmful interference.

## Markov Property

The observation vector is designed to include enough current information for decision-making: agent phase, battery, target direction, nearby agents, active orders, and local grid cells. This approximates the Markov property by reducing dependence on hidden history.
