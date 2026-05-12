# Reflection, Justification, and Original Thinking

## What Worked Well

1. **Auto-pickup and auto-drop** made the task learnable by reducing sparse credit assignment.
2. **Dueling Double DQN** is a good fit for grid navigation where many actions can have similar value.
3. **Streamlit dashboard** makes the system explainable and presentation-ready.
4. **Policy comparison** gives evidence beyond a single animated run.

## What Did Not Fully Work

The saved DQN checkpoint completes more deliveries than the baselines but has a high collision count in multi-seed evaluation. This means the agent learned to pursue task completion aggressively, but safety and coordination remain weak.

This is not a failure of the project; it is a meaningful research result. It shows why multi-agent warehouse routing is harder than single-agent path planning.

## Limitations

| Limitation | Impact | Future Fix |
|---|---|---|
| Independent learners | Agents can interfere with each other | QMIX or MAPPO |
| Local observation | Limited long-range planning | Graph or global observation |
| Uniform replay | Rare events under-sampled | Prioritized replay |
| Fixed layout | Limited generalization | Domain randomization |
| Collision penalty tuning | DQN may accept unsafe behavior | Stronger safety shaping |

## Original Extensions

Compared with a basic warehouse gridworld, TA-RWARE adds:

1. Battery dynamics and charging behavior.
2. Heterogeneous AGV/picker roles.
3. Auto-pickup/drop mechanics.
4. Dynamic order injection.
5. Dueling Double DQN architecture.
6. Interactive evaluation dashboard with deterministic policy comparison.

## What I Would Improve Next

The next strongest improvement would be centralized multi-agent training. A QMIX-style method could preserve decentralized execution while learning a joint value function, which should reduce collisions and improve coordination near goal docks.
