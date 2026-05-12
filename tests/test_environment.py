import unittest
from pathlib import Path

import numpy as np
import yaml

from envs.warehouse_env import AgentType, Phase, WarehouseEnv


ROOT = Path(__file__).resolve().parents[1]


def load_cfg():
    with open(ROOT / "configs" / "config.smoke.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class WarehouseEnvironmentTests(unittest.TestCase):
    def test_state_dim_and_agent_count(self):
        env = WarehouseEnv(load_cfg())
        obs, _ = env.reset(seed=0)
        self.assertEqual(len(obs), env.n_agents)
        for item in obs:
            self.assertEqual(item.shape, (env.state_dim,))

    def test_observations_stay_in_normalized_bounds(self):
        env = WarehouseEnv(load_cfg())
        obs, _ = env.reset(seed=0)
        for item in obs:
            self.assertTrue(np.all(item >= -1.0))
            self.assertTrue(np.all(item <= 1.0))

    def test_reset_seed_is_reproducible(self):
        env = WarehouseEnv(load_cfg())
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        for left, right in zip(obs1, obs2):
            self.assertTrue(np.allclose(left, right))
        self.assertEqual(info1["agent_positions"], info2["agent_positions"])

    def test_battery_drains_after_step(self):
        env = WarehouseEnv(load_cfg())
        env.reset(seed=0)
        initial = [agent.battery for agent in env.agents]
        env.step([4] * env.n_agents)
        final = [agent.battery for agent in env.agents]
        for before, after in zip(initial, final):
            self.assertLessEqual(after, before)

    def test_episode_truncates_at_max_steps(self):
        env = WarehouseEnv(load_cfg())
        env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps <= env.max_steps + 2:
            _, _, terminated, truncated, _ = env.step([4] * env.n_agents)
            done = terminated or truncated
            steps += 1
        self.assertTrue(done)
        self.assertLessEqual(steps, env.max_steps + 1)

    def test_auto_pickup_transition_when_agv_reaches_rack(self):
        env = WarehouseEnv(load_cfg())
        env.reset(seed=1)
        agv = next(agent for agent in env.agents if agent.type == AgentType.AGV and agent.order)
        agv.pos = [agv.order.rack_pos[0] - 1, agv.order.rack_pos[1]]
        agv.phase = Phase.TO_PICKUP
        _, rewards, _, _, _ = env.step([2 if agent.id == agv.id else 4 for agent in env.agents])
        self.assertEqual(agv.phase, Phase.TO_GOAL)
        self.assertGreaterEqual(rewards[agv.id], env.cfg["rewards"]["pickup_item"] + env.cfg["rewards"]["time_step"])


if __name__ == "__main__":
    unittest.main()
