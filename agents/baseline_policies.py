from __future__ import annotations

import numpy as np

from envs.warehouse_env import AgentType, Cell, Phase


WAIT_ACTION = 4


def _nearest_charger(env, agent):
    return min(
        env.charge_cells,
        key=lambda c: abs(c[0] - agent.pos[0]) + abs(c[1] - agent.pos[1])
    )


def _agent_target(env, agent):
    if agent.battery / agent.batt_cap < env.batt_lo:
        return _nearest_charger(env, agent)
    if agent.type == AgentType.PICKER:
        return env._picker_target(agent)
    if agent.order and not agent.order.done:
        if agent.phase == Phase.TO_GOAL:
            return tuple(agent.order.goal_pos)
        return tuple(agent.order.rack_pos)
    return None


def _valid_move(env, agent, dest, occupied):
    nc, nr = dest
    if not (0 <= nc < env.W and 0 <= nr < env.H):
        return False
    if dest in (occupied - {tuple(agent.pos)}):
        return False
    cell = env.grid[nr, nc]
    if cell == Cell.WALL:
        return False
    if cell == Cell.RACK:
        is_own_rack = (
            agent.type == AgentType.AGV and
            agent.order and
            not agent.order.done and
            agent.phase == Phase.TO_PICKUP and
            [nc, nr] == agent.order.rack_pos
        )
        return is_own_rack
    return True


def greedy_actions(env):
    actions = []
    occupied = {tuple(a.pos) for a in env.agents}

    for agent in env.agents:
        target = _agent_target(env, agent)
        if target is None:
            actions.append(WAIT_ACTION)
            continue

        best_action = WAIT_ACTION
        best_score = abs(target[0] - agent.pos[0]) + abs(target[1] - agent.pos[1])
        candidates = list(env.MOVES.items())
        candidates.sort(key=lambda item: item[0])

        for act, (dc, dr) in candidates:
            dest = (agent.pos[0] + dc, agent.pos[1] + dr)
            if not _valid_move(env, agent, dest, occupied):
                continue
            score = abs(target[0] - dest[0]) + abs(target[1] - dest[1])
            if score < best_score:
                best_score = score
                best_action = act

        actions.append(best_action)

    return actions


def random_actions(env, rng=None):
    rng = rng or np.random.default_rng()
    return [int(rng.integers(env.action_dim)) for _ in env.agents]
