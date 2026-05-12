"""
TA-RWARE Final - Warehouse Environment
========================================
ROOT CAUSE FIX:
  The previous version required agents to BOTH navigate to rack AND
  press a separate ACT action. The DQN almost never learned to do both
  in sequence because the reward signal was too sparse.

SOLUTION:
  Auto-pickup: when AGV steps onto its target rack -> item picked up automatically
  Auto-drop:   when AGV steps onto its target goal -> item delivered automatically
  Actions are now just: N / S / E / W / WAIT
  This means the DQN only needs to learn navigation, not a complex sequence.
  Delivery reward is immediate and dense.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List
from enum import IntEnum


class Cell(IntEnum):
    EMPTY  = 0
    WALL   = 1
    RACK   = 2
    GOAL   = 3
    CHARGE = 4


class AgentType(IntEnum):
    AGV    = 0
    PICKER = 1


class Phase(IntEnum):
    IDLE      = 0
    TO_PICKUP = 1   # heading to rack
    TO_GOAL   = 2   # carrying to goal dock
    CHARGING  = 3


class Order:
    _ctr = 0

    def __init__(self, shelf_id, rack_pos, goal_pos, created_step):
        Order._ctr += 1
        self.id       = Order._ctr
        self.shelf_id = shelf_id
        self.rack_pos = list(rack_pos)
        self.goal_pos = list(goal_pos)
        self.agv_id   = -1
        self.done     = False
        self.created_step  = int(created_step)
        self.assigned_step = None
        self.pickup_step   = None
        self.completed_step = None


class Agent:
    def __init__(self, aid, atype, pos, batt_cap):
        self.id        = aid
        self.type      = atype
        self.pos       = list(pos)
        self.phase     = Phase.IDLE
        self.order     = None
        self.battery   = float(batt_cap)
        self.batt_cap  = float(batt_cap)
        self.charging  = False
        self.progress_key = None
        self.best_dist_to_target = None
        # metrics
        self.deliveries  = 0
        self.assists     = 0
        self.steps_moved = 0
        self.collisions  = 0
        self.wait_steps  = 0
        self.charging_steps = 0
        self.charge_visits = 0
        self.empty_events = 0


class WarehouseEnv(gym.Env):
    """
    Fast warehouse environment with AUTO pickup/drop on cell arrival.

    Actions (Discrete 5):
      0=NORTH  1=SOUTH  2=EAST  3=WEST  4=WAIT

    Pickup happens automatically when AGV steps on its rack target.
    Drop    happens automatically when AGV steps on its goal dock.
    This removes the need for a separate ACT action and makes the
    reward signal dense and immediate.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 8}

    # Direction vectors
    MOVES = {0: (0,-1), 1: (0,1), 2: (1,0), 3: (-1,0)}

    def __init__(self, config: dict, render_mode=None):
        super().__init__()
        self.cfg         = config
        self.render_mode = render_mode
        env_cfg          = config['environment']

        self.n_rows   = env_cfg['grid_rows']
        self.n_cols   = env_cfg['grid_cols']
        self.n_agvs   = env_cfg['n_agvs']
        self.n_pick   = env_cfg['n_pickers']
        self.n_agents = self.n_agvs + self.n_pick
        self.max_steps = env_cfg['max_steps']
        self.req_q    = env_cfg['request_queue_size']
        self.inj_int  = env_cfg['task_injection_interval']
        self.pick_radius = env_cfg.get('picker_service_radius', 1)
        self.batt_cap = env_cfg['battery_capacity']
        self.batt_lo  = env_cfg.get('battery_low_threshold', 0.20)
        self.bd_move  = env_cfg['battery_drain_move']
        self.bd_idle  = env_cfg['battery_drain_idle']
        self.b_rech   = env_cfg['battery_recharge_rate']

        agent_cfg = config['agent']
        self.local_view_radius = agent_cfg.get('local_view_radius', 2)
        self.max_visible_orders = agent_cfg.get(
            'max_visible_pending_orders',
            min(4, self.req_q)
        )
        self.max_visible_agents = agent_cfg.get(
            'max_visible_other_agents',
            min(6, max(0, self.n_agents - 1))
        )
        self.state_dim  = self._expected_state_dim()
        self.action_dim = config['agent']['action_dim']

        self._build_grid()

        self.observation_space = spaces.Box(
            -1.0, 1.0, (self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)

        self.agents: List[Agent] = []
        self.orders: List[Order] = []
        self.total_delivered     = 0
        self.steps               = 0
        self.total_reward        = 0.0
        self._pg                 = None

        self.reset()

    def _expected_state_dim(self):
        local_side = self.local_view_radius * 2 + 1
        return (
            5 +                       # own state
            3 +                       # target direction + distance
            4 +                       # task / helper coordinates
            self.max_visible_orders * 2 +
            self.max_visible_agents * 3 +
            local_side * local_side
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Grid construction
    # ─────────────────────────────────────────────────────────────────────────
    def _build_grid(self):
        shelf_h = 2
        aisle_h = 1
        pad     = 1
        H = pad*2 + self.n_rows*(shelf_h + aisle_h) + aisle_h + 2
        W = self.n_cols*2 + 4

        self.H, self.W = H, W
        self.grid = np.zeros((H, W), dtype=np.int8)

        # Perimeter walls
        self.grid[0, :]  = Cell.WALL
        self.grid[-1, :] = Cell.WALL
        self.grid[:, 0]  = Cell.WALL
        self.grid[:, -1] = Cell.WALL

        # Rack cells at even columns (2, 4, 6, ...)
        self.rack_cells = []
        self.shelf_ids  = {}
        sid = 0
        for rband in range(self.n_rows):
            r0 = pad + 1 + aisle_h + rband*(shelf_h + aisle_h)
            for sh in range(shelf_h):
                row = r0 + sh
                for rcol in range(self.n_cols):
                    col = 2 + rcol*2
                    if 0 < row < H-1 and 0 < col < W-1:
                        self.grid[row, col] = Cell.RACK
                        self.rack_cells.append((col, row))
                        self.shelf_ids[(col, row)] = sid
                        sid += 1

        self.n_shelves = sid

        # Goal docks on rightmost interior column
        self.goal_cells = []
        gcol = W - 2
        for row in range(1, H-1):
            if self.grid[row, gcol] == Cell.EMPTY:
                self.grid[row, gcol] = Cell.GOAL
                self.goal_cells.append((gcol, row))

        # Charging stations distributed along the left aisle for larger fleets
        self.charge_cells = []
        charge_target = max(4, self.n_agvs)
        candidates = [
            (c, r)
            for c in range(1, min(4, W - 1))
            for r in range(1, H - 1)
            if self.grid[r, c] == Cell.EMPTY
        ]
        if candidates:
            idxs = np.linspace(
                0, len(candidates) - 1,
                num=min(charge_target, len(candidates)),
                dtype=int
            )
            seen = set()
            for idx in idxs:
                cell = candidates[int(idx)]
                if cell in seen:
                    continue
                seen.add(cell)
                c, r = cell
                self.grid[r, c] = Cell.CHARGE
                self.charge_cells.append(cell)

    # ─────────────────────────────────────────────────────────────────────────
    # Order management
    # ─────────────────────────────────────────────────────────────────────────
    def _free_aisle_cells(self):
        cells = []
        for r in range(1, self.H-1):
            for c in range(1, self.W-1):
                if self.grid[r, c] == Cell.EMPTY:
                    cells.append((c, r))
        return cells

    def _new_order(self):
        used  = {tuple(o.rack_pos) for o in self.orders if not o.done}
        avail = [p for p in self.rack_cells if tuple(p) not in used]
        if not avail:
            return
        rack = list(avail[np.random.randint(len(avail))])
        goal = list(self.goal_cells[np.random.randint(len(self.goal_cells))])
        sid  = self.shelf_ids[tuple(rack)]
        self.orders.append(Order(sid, rack, goal, self.steps))

    def _assign_orders(self):
        """Assign nearest pending order to each idle AGV."""
        idle = [a for a in self.agents
                if a.type == AgentType.AGV
                and a.phase == Phase.IDLE
                and not a.charging]
        pend = [o for o in self.orders
                if not o.done and o.agv_id == -1]

        for agv in idle:
            if not pend:
                break
            # Nearest-first assignment
            best = min(pend,
                key=lambda o: (abs(o.rack_pos[0] - agv.pos[0])
                             + abs(o.rack_pos[1] - agv.pos[1])))
            best.agv_id = agv.id
            if best.assigned_step is None:
                best.assigned_step = self.steps
            agv.order   = best
            agv.phase   = Phase.TO_PICKUP
            pend.remove(best)

    def _picker_target(self, agent):
        active_goals = []
        for ag in self.agents:
            if (ag.type == AgentType.AGV and ag.order and not ag.order.done
                    and ag.phase in (Phase.TO_PICKUP, Phase.TO_GOAL)):
                active_goals.append(
                    ag.order.goal_pos if ag.phase == Phase.TO_GOAL else ag.order.rack_pos
                )

        if active_goals:
            return min(
                active_goals,
                key=lambda tgt: abs(tgt[0] - agent.pos[0]) + abs(tgt[1] - agent.pos[1])
            )

        return min(
            self.goal_cells,
            key=lambda tgt: abs(tgt[0] - agent.pos[0]) + abs(tgt[1] - agent.pos[1])
        )

    def _target_key(self, agent):
        if agent.type == AgentType.PICKER:
            tgt = self._picker_target(agent)
            return ('picker', tgt[0], tgt[1])
        if not agent.order or agent.order.done:
            return None
        if agent.phase == Phase.TO_PICKUP:
            tgt = agent.order.rack_pos
            return ('rack', agent.order.id, tgt[0], tgt[1])
        if agent.phase == Phase.TO_GOAL:
            tgt = agent.order.goal_pos
            return ('goal', agent.order.id, tgt[0], tgt[1])
        return None

    def _refresh_progress_anchor(self, agent, current_dist):
        key = self._target_key(agent)
        if key != agent.progress_key:
            agent.progress_key = key
            agent.best_dist_to_target = current_dist
        elif current_dist is not None and agent.best_dist_to_target is None:
            agent.best_dist_to_target = current_dist

    def _select_charger(self, agent, occupied, charger_claims):
        free_chargers = []
        queued_chargers = []
        for charger in self.charge_cells:
            dist = abs(charger[0] - agent.pos[0]) + abs(charger[1] - agent.pos[1])
            score = dist + charger_claims.get(charger, 0) * 4
            if charger not in (occupied - {tuple(agent.pos)}):
                free_chargers.append((score, dist, charger))
            else:
                queued_chargers.append((score + 6, dist, charger))

        pool = free_chargers if free_chargers else queued_chargers
        _, _, charger = min(pool, key=lambda item: (item[0], item[1]))
        charger_claims[charger] = charger_claims.get(charger, 0) + 1
        return charger

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        Order._ctr = 0

        free = self._free_aisle_cells()
        np.random.shuffle(free)

        self.agents = []
        for i in range(self.n_agvs):
            pos = list(free[i % len(free)])
            self.agents.append(Agent(i, AgentType.AGV, pos, self.batt_cap))
        for j in range(self.n_pick):
            pos = list(free[(self.n_agvs + j) % len(free)])
            self.agents.append(
                Agent(self.n_agvs + j, AgentType.PICKER, pos, self.batt_cap)
            )

        self.orders          = []
        self.total_delivered = 0
        self.steps           = 0
        self.total_reward    = 0.0

        for _ in range(self.req_q):
            self._new_order()
        self._assign_orders()

        obs  = [self._observe(i) for i in range(self.n_agents)]
        info = self._info()
        return obs, info

    # ─────────────────────────────────────────────────────────────────────────
    # Observation (dynamic dims from config)
    # ─────────────────────────────────────────────────────────────────────────
    def _observe(self, idx):
        a   = self.agents[idx]
        obs = np.zeros(self.state_dim, dtype=np.float32)
        p   = 0

        def put(*vals):
            nonlocal p
            for v in vals:
                if p < self.state_dim:
                    obs[p] = float(np.clip(v, -1.0, 1.0))
                    p += 1

        carrying = (a.phase == Phase.TO_GOAL)
        picker_target = self._picker_target(a) if a.type == AgentType.PICKER else None

        # Own state (5 dims)
        put(a.pos[0] / self.W,
            a.pos[1] / self.H,
            a.battery / a.batt_cap,
            1.0 if carrying else 0.0,
            a.phase / 3.0)

        # Target direction & distance (3 dims)
        if a.order and not a.order.done:
            tgt = a.order.goal_pos if carrying else a.order.rack_pos
        elif picker_target is not None:
            tgt = picker_target
        else:
            tgt = None

        if tgt is not None:
            put((tgt[0] - a.pos[0]) / self.W,
                (tgt[1] - a.pos[1]) / self.H,
                (abs(tgt[0]-a.pos[0]) + abs(tgt[1]-a.pos[1])) / (self.W+self.H))
        else:
            put(0.0, 0.0, 0.0)

        # Order rack + goal coords (4 dims)
        if a.order and not a.order.done:
            put(a.order.rack_pos[0] / self.W,
                a.order.rack_pos[1] / self.H,
                a.order.goal_pos[0] / self.W,
                a.order.goal_pos[1] / self.H)
        elif picker_target is not None:
            nearest_charge = min(
                self.charge_cells,
                key=lambda c: abs(c[0] - a.pos[0]) + abs(c[1] - a.pos[1])
            )
            put(picker_target[0] / self.W,
                picker_target[1] / self.H,
                nearest_charge[0] / self.W,
                nearest_charge[1] / self.H)
        else:
            put(0.0, 0.0, 0.0, 0.0)

        # Pending orders (configurable count)
        pend = [o for o in self.orders
                if not o.done and o.agv_id == -1][:self.max_visible_orders]
        for o in pend:
            put(o.rack_pos[0] / self.W, o.rack_pos[1] / self.H)
        put(*([0.0] * max(0, self.max_visible_orders * 2 - len(pend)*2)))

        # Closest other agents (configurable count)
        others = sorted(
            [ag for ag in self.agents if ag.id != idx],
            key=lambda ag: abs(ag.pos[0] - a.pos[0]) + abs(ag.pos[1] - a.pos[1])
        )[:self.max_visible_agents]
        for o in others:
            put((o.pos[0] - a.pos[0]) / self.W,
                (o.pos[1] - a.pos[1]) / self.H,
                1.0 if o.phase == Phase.TO_GOAL else 0.0)
        put(*([0.0] * max(0, self.max_visible_agents * 3 - len(others)*3)))

        # Local grid perception
        for dr in range(-self.local_view_radius, self.local_view_radius + 1):
            for dc in range(-self.local_view_radius, self.local_view_radius + 1):
                nc, nr = a.pos[0]+dc, a.pos[1]+dr
                if 0 <= nc < self.W and 0 <= nr < self.H:
                    cv = float(self.grid[nr, nc])
                    # Mark other agents
                    for ag2 in self.agents:
                        if ag2.id != idx and ag2.pos == [nc, nr]:
                            cv = 5.0
                    put(cv / 5.0)
                else:
                    put(0.2)   # treat OOB as wall

        return obs[:self.state_dim]

    # ─────────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, actions):
        self.steps += 1
        rewards  = [0.0] * self.n_agents
        occupied = {tuple(a.pos) for a in self.agents}
        charger_claims = {}

        # Dynamic order injection
        if self.steps % self.inj_int == 0:
            self._new_order()
            self._assign_orders()

        for i, agent in enumerate(self.agents):
            act = int(actions[i]) if i < len(actions) else 4  # default WAIT

            rewards[i] += self.cfg['rewards']['time_step']

            # ── Battery: recharging ───────────────────────────────────────
            if agent.charging:
                agent.charging_steps += 1
                agent.battery = min(agent.batt_cap,
                                    agent.battery + self.b_rech)
                if agent.battery >= agent.batt_cap:
                    agent.charging = False
                    agent.phase    = Phase.IDLE
                    self._assign_orders()
                continue

            # ── Battery: empty ─────────────────────────────────────────────
            if agent.battery <= 0:
                rewards[i] += self.cfg['rewards']['battery_empty']
                agent.empty_events += 1
                agent.battery = 1.0   # give tiny amount to move
                # Force to nearest charger
                nearest = self._select_charger(agent, occupied, charger_claims)
                self._step_toward(agent, nearest, occupied)
                if tuple(agent.pos) == tuple(nearest):
                    agent.charging = True
                    agent.phase    = Phase.CHARGING
                    agent.charge_visits += 1
                continue

            # ── Battery: low → head to charger ─────────────────────────────
            if agent.battery / agent.batt_cap < self.batt_lo:
                nearest = self._select_charger(agent, occupied, charger_claims)
                moved = self._step_toward(agent, nearest, occupied)
                agent.battery -= self.bd_move if moved else self.bd_idle
                if tuple(agent.pos) == tuple(nearest):
                    agent.charging = True
                    agent.phase    = Phase.CHARGING
                    agent.charge_visits += 1
                continue

            # ── Normal movement ────────────────────────────────────────────
            if act in self.MOVES:
                dc, dr = self.MOVES[act]
                nc, nr = agent.pos[0]+dc, agent.pos[1]+dr

                # Compute distance to target BEFORE moving (for shaping)
                old_dist = self._dist_to_target(agent)
                self._refresh_progress_anchor(agent, old_dist)
                old_progress_key = agent.progress_key

                # Check if destination is valid
                valid_move = (
                    0 <= nc < self.W and
                    0 <= nr < self.H and
                    (nc, nr) not in (occupied - {tuple(agent.pos)})
                )

                # Racks: only AGV can enter its own target rack
                if valid_move and self.grid[nr, nc] == Cell.RACK:
                    is_own_rack = (
                        agent.type == AgentType.AGV and
                        agent.order and
                        not agent.order.done and
                        agent.phase == Phase.TO_PICKUP and
                        [nc, nr] == agent.order.rack_pos
                    )
                    if not is_own_rack:
                        valid_move = False

                if valid_move:
                    occupied.discard(tuple(agent.pos))
                    agent.pos = [nc, nr]
                    occupied.add((nc, nr))
                    agent.battery    -= self.bd_move
                    agent.steps_moved += 1

                    # ── AUTO PICKUP: arrived at rack ───────────────────────
                    if (agent.type == AgentType.AGV and
                            agent.order and
                            not agent.order.done and
                            agent.phase == Phase.TO_PICKUP and
                            agent.pos == agent.order.rack_pos):
                        agent.phase   = Phase.TO_GOAL
                        agent.order.pickup_step = self.steps
                        rewards[i]   += self.cfg['rewards']['pickup_item']

                    # ── AUTO DROP: arrived at goal ─────────────────────────
                    elif (agent.type == AgentType.AGV and
                              agent.order and
                              not agent.order.done and
                              agent.phase == Phase.TO_GOAL and
                              agent.pos == agent.order.goal_pos):
                        agent.order.done  = True
                        agent.order.completed_step = self.steps
                        self.total_delivered += 1
                        agent.deliveries  += 1
                        rewards[i]        += self.cfg['rewards']['delivery_complete']
                        agent.phase        = Phase.IDLE
                        agent.order        = None
                        # Picker assist bonus only for nearby dock workers.
                        for pk in self.agents:
                            if (pk.type == AgentType.PICKER and
                                    abs(pk.pos[0] - agent.pos[0]) + abs(pk.pos[1] - agent.pos[1]) <= self.pick_radius):
                                pk.assists += 1
                                rewards[pk.id] += self.cfg['rewards'].get('picker_assist', 2.0)
                        self._assign_orders()

                    # ── Progress shaping ───────────────────────────────────
                    new_dist = self._dist_to_target(agent)
                    new_progress_key = self._target_key(agent)
                    if new_progress_key != old_progress_key:
                        agent.progress_key = new_progress_key
                        agent.best_dist_to_target = new_dist
                    elif new_dist is not None:
                        best_dist = agent.best_dist_to_target
                        if best_dist is None:
                            agent.best_dist_to_target = new_dist
                            best_dist = new_dist

                        if new_dist < best_dist:
                            rewards[i] += self.cfg['rewards']['progress'] * (best_dist - new_dist)
                            agent.best_dist_to_target = new_dist
                        elif old_dist is not None and new_dist > old_dist:
                            rewards[i] += self.cfg['rewards']['move_away'] * (new_dist - old_dist)

                else:
                    # Collision with wall / other agent / invalid rack
                    rewards[i]       += self.cfg['rewards']['collision']
                    agent.battery    -= self.bd_idle
                    agent.collisions += 1

            else:   # WAIT
                agent.battery -= self.bd_idle
                agent.wait_steps += 1

        # ── Team completion bonus ─────────────────────────────────────────
        terminated = all(o.done for o in self.orders) and len(self.orders) > 0
        if terminated:
            for i in range(self.n_agents):
                rewards[i] += self.cfg['rewards']['team_bonus']

        truncated = (self.steps >= self.max_steps)
        self.total_reward += sum(rewards)

        obs  = [self._observe(i) for i in range(self.n_agents)]
        info = self._info()
        info['rewards'] = rewards
        return obs, rewards, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _dist_to_target(self, agent):
        """Manhattan distance to agent's current navigation target."""
        if agent.type == AgentType.PICKER:
            tgt = self._picker_target(agent)
            return abs(agent.pos[0]-tgt[0]) + abs(agent.pos[1]-tgt[1])
        if not agent.order or agent.order.done:
            return None
        if agent.phase == Phase.TO_PICKUP:
            tgt = agent.order.rack_pos
        elif agent.phase == Phase.TO_GOAL:
            tgt = agent.order.goal_pos
        else:
            return None
        return abs(agent.pos[0]-tgt[0]) + abs(agent.pos[1]-tgt[1])

    def _step_toward(self, agent, target, occupied):
        """Move agent one step toward target (used for battery emergency)."""
        dx = np.sign(target[0] - agent.pos[0])
        dy = np.sign(target[1] - agent.pos[1])
        for dc, dr in [(dx, 0), (0, dy), (dx, dy)]:
            if dc == 0 and dr == 0:
                continue
            nc, nr = agent.pos[0]+int(dc), agent.pos[1]+int(dr)
            if (0 <= nc < self.W and 0 <= nr < self.H and
                    self.grid[nr, nc] not in (Cell.WALL, Cell.RACK) and
                    (nc, nr) not in (occupied - {tuple(agent.pos)})):
                occupied.discard(tuple(agent.pos))
                agent.pos = [nc, nr]
                occupied.add((nc, nr))
                agent.steps_moved += 1
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Info
    # ─────────────────────────────────────────────────────────────────────────
    def _info(self):
        return {
            'steps':           self.steps,
            'total_reward':    self.total_reward,
            'deliveries':      self.total_delivered,
            'total_orders':    len(self.orders),
            'pending_orders':  sum(1 for o in self.orders
                                   if not o.done and o.agv_id == -1),
            'battery_levels':  [a.battery for a in self.agents],
            'agent_positions': [a.pos[:] for a in self.agents],
            'agent_metrics': [{
                'deliveries': a.deliveries,
                'assists':    a.assists,
                'distance':   a.steps_moved,
                'collisions': a.collisions,
                'wait_steps': a.wait_steps,
                'charging_steps': a.charging_steps,
                'charge_visits': a.charge_visits,
                'empty_events': a.empty_events,
            } for a in self.agents],
            'completed_order_times': [
                int(o.completed_step - o.created_step)
                for o in self.orders
                if o.done and o.completed_step is not None
            ],
        }

    def close(self):
        if self._pg:
            import pygame
            pygame.quit()
