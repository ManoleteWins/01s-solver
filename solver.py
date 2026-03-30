from __future__ import annotations
import numpy as np
from multiprocessing import Process, Value, shared_memory
from typing import Dict, List, Tuple, Callable, Optional
from config import GameConfig
from game import GameState, Action


def _shared_worker(config, initial_reach, node_locks, strategy_locks,
                   index_map, n_actions_map, D, max_actions,
                   regret_shm_name, strategy_shm_name, arr_shape,
                   iterations, progress_counter):
    """Worker: run CFR iterations reading/writing shared memory arrays."""
    regret_shm = shared_memory.SharedMemory(name=regret_shm_name)
    strategy_shm = shared_memory.SharedMemory(name=strategy_shm_name)
    regret_arr = np.ndarray(arr_shape, dtype=np.float64, buffer=regret_shm.buf)
    strategy_arr = np.ndarray(arr_shape, dtype=np.float64, buffer=strategy_shm.buf)

    N = config.num_players

    for _ in range(iterations):
        for p in range(N):
            root = GameState(config)
            reach = [r.copy() for r in initial_reach]
            _cfr_shared(root, reach, p, D, N, regret_arr, strategy_arr,
                        index_map, n_actions_map, node_locks, strategy_locks,
                        initial_reach)
        with progress_counter.get_lock():
            progress_counter.value += 1

    regret_shm.close()
    strategy_shm.close()


def _cfr_shared(state, reach, traverser, D, N, regret_arr, strategy_arr,
                index_map, n_actions_map, node_locks, strategy_locks,
                initial_reach):
    """CFR traversal using shared memory arrays."""
    if state.is_terminal():
        return _terminal_shared(state, reach, traverser, D, N)

    player = state.current_player()
    key = (player, state.history)
    idx = index_map[key]
    num_actions = n_actions_map[key]

    # Get strategy
    if key in strategy_locks:
        strategy = strategy_locks[key]
    else:
        regrets = np.maximum(regret_arr[idx, :, :num_actions].copy(), 0.0)
        total = regrets.sum(axis=1, keepdims=True)
        strategy = np.full((D, num_actions), 1.0 / num_actions)
        mask = (total > 0).flatten()
        if mask.any():
            strategy[mask] = regrets[mask] / total[mask]

        if key in node_locks:
            strategy = _project_shared(strategy, reach[player],
                                       node_locks[key], num_actions, D)

    if player == traverser:
        actions = state.available_actions()
        action_values = np.zeros((num_actions, D))

        for a_idx in range(num_actions):
            new_state = state.apply(actions[a_idx])
            new_reach = [r.copy() for r in reach]
            new_reach[traverser] = new_reach[traverser] * strategy[:, a_idx]
            action_values[a_idx] = _cfr_shared(
                new_state, new_reach, traverser, D, N,
                regret_arr, strategy_arr, index_map, n_actions_map,
                node_locks, strategy_locks, initial_reach)

        node_value = np.einsum('ad,da->d', action_values, strategy)

        if key not in strategy_locks:
            # Write directly to shared memory (benign races)
            for a_idx in range(num_actions):
                regret_arr[idx, :, a_idx] += action_values[a_idx] - node_value
            strategy_arr[idx, :, :num_actions] += reach[player][:, np.newaxis] * strategy

        return node_value
    else:
        actions = state.available_actions()
        total_value = np.zeros(D)
        for a_idx in range(num_actions):
            new_state = state.apply(actions[a_idx])
            new_reach = [r.copy() for r in reach]
            new_reach[player] = new_reach[player] * strategy[:, a_idx]
            total_value += _cfr_shared(
                new_state, new_reach, traverser, D, N,
                regret_arr, strategy_arr, index_map, n_actions_map,
                node_locks, strategy_locks, initial_reach)
        return total_value


def _terminal_shared(state, reach, traverser, D, N):
    """Terminal value using CDF trick."""
    active = state.active_players()
    t = traverser

    R = 1.0
    for j in range(N):
        if j != t:
            R *= np.sum(reach[j])

    if t not in active:
        return np.full(D, -state.contributions[t] * R)

    active_opponents = [p for p in active if p != t]

    if not active_opponents:
        return np.full(D, (state.pot - state.contributions[t]) * R)

    F = 1.0
    for j in range(N):
        if j != t and j not in active:
            F *= np.sum(reach[j])

    win_weight = np.ones(D)
    for opp in active_opponents:
        cdf = np.cumsum(reach[opp])
        cdf_shifted = np.zeros(D)
        cdf_shifted[1:] = cdf[:-1]
        win_weight *= cdf_shifted

    return state.pot * F * win_weight - state.contributions[t] * R


def _project_shared(strategy, reach, locks, num_actions, D):
    """Project strategy to meet aggregate frequency constraints."""
    proj = strategy.copy()
    total_reach = reach.sum()
    if total_reach < 1e-15:
        return proj
    weights = reach / total_reach
    for _ in range(20):
        for a_idx, target in locks.items():
            current = (weights * proj[:, a_idx]).sum()
            if current > 1e-15:
                proj[:, a_idx] *= target / current
            elif target > 0:
                in_range = reach > 1e-15
                if in_range.any():
                    proj[in_range, a_idx] = target / in_range.sum()
        proj = np.maximum(proj, 0.0)
        row_sums = proj.sum(axis=1, keepdims=True)
        proj = np.where(row_sums > 1e-15, proj / row_sums, 1.0 / num_actions)
    return proj


class CFRSolver:
    """Vanilla CFR solver for N-player [0,1] games."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.D = config.discretization
        self.N = config.num_players
        self.hand_values = np.linspace(0.5 / self.D, 1.0 - 0.5 / self.D, self.D)

        self.initial_reach: List[np.ndarray] = []
        for p in range(self.N):
            reach = np.zeros(self.D)
            rc = config.range_configs[p]
            for i, val in enumerate(self.hand_values):
                if rc.contains(val):
                    reach[i] = 1.0 / self.D
            self.initial_reach.append(reach)

        self.info_sets: Dict[Tuple, List[Action]] = {}
        self.node_locks: Dict[Tuple, Dict[int, float]] = {}
        self.strategy_locks: Dict[Tuple, np.ndarray] = {}
        self.iterations_done = 0

        # Array-based storage (populated by _discover_tree)
        self._index_map: Optional[Dict[Tuple, int]] = None
        self._n_actions_map: Optional[Dict[Tuple, int]] = None
        self._max_actions = 0
        self._n_info_sets = 0
        self.regret_data: Optional[np.ndarray] = None   # (n_info_sets, D, max_actions)
        self.strategy_data: Optional[np.ndarray] = None

    # ── Tree discovery ──────────────────────────────────────────

    def _discover_tree(self):
        """Traverse game tree once to find all info sets."""
        self.info_sets = {}
        self._discover(GameState(self.config))

        sorted_keys = sorted(self.info_sets.keys())
        self._index_map = {k: i for i, k in enumerate(sorted_keys)}
        self._n_actions_map = {k: len(self.info_sets[k]) for k in sorted_keys}
        self._n_info_sets = len(sorted_keys)
        self._max_actions = max(len(a) for a in self.info_sets.values()) if self.info_sets else 1

        shape = (self._n_info_sets, self.D, self._max_actions)
        self.regret_data = np.zeros(shape)
        self.strategy_data = np.zeros(shape)

    def _discover(self, state: GameState):
        if state.is_terminal():
            return
        player = state.current_player()
        actions = state.available_actions()
        key = (player, state.history)
        if key not in self.info_sets:
            self.info_sets[key] = actions
            for action in actions:
                self._discover(state.apply(action))

    def _ensure_discovered(self):
        if self._index_map is None:
            self._discover_tree()

    # ── Strategy access ─────────────────────────────────────────

    def _get_strategy(self, player: int, history: tuple, num_actions: int,
                       reach: Optional[np.ndarray] = None) -> np.ndarray:
        key = (player, history)
        if key in self.strategy_locks:
            return self.strategy_locks[key]

        idx = self._index_map[key]
        regrets = np.maximum(self.regret_data[idx, :, :num_actions].copy(), 0.0)
        total = regrets.sum(axis=1, keepdims=True)
        strategy = np.full((self.D, num_actions), 1.0 / num_actions)
        mask = (total > 0).flatten()
        if mask.any():
            strategy[mask] = regrets[mask] / total[mask]

        if key in self.node_locks and reach is not None:
            strategy = self._project_to_freq(strategy, reach, self.node_locks[key], num_actions)
        return strategy

    def get_average_strategy(self, player: int, history: tuple, num_actions: int) -> np.ndarray:
        key = (player, history)
        if key in self.strategy_locks:
            return self.strategy_locks[key]

        self._ensure_discovered()
        idx = self._index_map[key]
        data = self.strategy_data[idx, :, :num_actions]
        total = data.sum(axis=1, keepdims=True)
        avg = np.where(total > 0, data / total, 1.0 / num_actions)

        if key in self.node_locks:
            avg = self._project_to_freq(avg, self.initial_reach[player],
                                        self.node_locks[key], num_actions)
        return avg

    # ── Training ────────────────────────────────────────────────

    def train(self, iterations: int, callback: Optional[Callable[[int, int], None]] = None):
        """Single-threaded vanilla CFR."""
        self._ensure_discovered()
        for t in range(iterations):
            for p in range(self.N):
                root = GameState(self.config)
                reach = [r.copy() for r in self.initial_reach]
                self._cfr(root, reach, p)
            self.iterations_done += 1
            if callback:
                callback(self.iterations_done, iterations)

    def train_parallel(self, iterations: int, num_workers: int = 4,
                       callback: Optional[Callable[[int, int], None]] = None):
        """Parallel CFR using shared memory. Workers read/write the same
        regret and strategy arrays concurrently (benign races, like PioSOLVER)."""
        if num_workers <= 1:
            return self.train(iterations, callback)

        self._ensure_discovered()
        shape = self.regret_data.shape
        nbytes = int(np.prod(shape) * 8)

        # Create shared memory and copy current data
        regret_shm = shared_memory.SharedMemory(create=True, size=nbytes)
        strategy_shm = shared_memory.SharedMemory(create=True, size=nbytes)
        regret_shared = np.ndarray(shape, dtype=np.float64, buffer=regret_shm.buf)
        strategy_shared = np.ndarray(shape, dtype=np.float64, buffer=strategy_shm.buf)
        np.copyto(regret_shared, self.regret_data)
        np.copyto(strategy_shared, self.strategy_data)

        # Shared progress counter (total iterations completed across all workers)
        progress = Value('i', 0)
        self._base_iterations = self.iterations_done

        # Split iterations across workers
        per_worker = []
        left = iterations
        for w in range(num_workers):
            n = left // (num_workers - w)
            per_worker.append(n)
            left -= n

        # Spawn workers
        processes = []
        for n in per_worker:
            if n <= 0:
                continue
            p = Process(target=_shared_worker, args=(
                self.config, self.initial_reach, self.node_locks, self.strategy_locks,
                self._index_map, self._n_actions_map, self.D, self._max_actions,
                regret_shm.name, strategy_shm.name, shape, n, progress))
            p.start()
            processes.append(p)

        # Poll progress while workers run
        import time
        while any(p.is_alive() for p in processes):
            time.sleep(0.1)
            done = progress.value
            self.iterations_done = self._base_iterations + done
            if callback:
                callback(done, iterations)

        for p in processes:
            p.join()

        # Final progress update
        self.iterations_done = self._base_iterations + iterations
        if callback:
            callback(iterations, iterations)

        # Copy results back
        np.copyto(self.regret_data, regret_shared)
        np.copyto(self.strategy_data, strategy_shared)

        # Cleanup
        regret_shm.close()
        regret_shm.unlink()
        strategy_shm.close()
        strategy_shm.unlink()

    # ── CFR traversal (single-thread, uses self arrays) ─────────

    def _cfr(self, state: GameState, reach: List[np.ndarray], traverser: int) -> np.ndarray:
        if state.is_terminal():
            return self._terminal_value(state, reach, traverser)

        player = state.current_player()
        actions = state.available_actions()
        num_actions = len(actions)
        key = (player, state.history)

        if key not in self.info_sets:
            self.info_sets[key] = actions

        strategy = self._get_strategy(player, state.history, num_actions, reach[player])

        if player == traverser:
            action_values = np.zeros((num_actions, self.D))
            for a_idx, action in enumerate(actions):
                new_state = state.apply(action)
                new_reach = [r.copy() for r in reach]
                new_reach[traverser] = new_reach[traverser] * strategy[:, a_idx]
                action_values[a_idx] = self._cfr(new_state, new_reach, traverser)

            node_value = np.einsum('ad,da->d', action_values, strategy)

            if key not in self.strategy_locks:
                idx = self._index_map[key]
                for a_idx in range(num_actions):
                    self.regret_data[idx, :, a_idx] += action_values[a_idx] - node_value
                self.strategy_data[idx, :, :num_actions] += reach[player][:, np.newaxis] * strategy

            return node_value
        else:
            total_value = np.zeros(self.D)
            for a_idx, action in enumerate(actions):
                new_state = state.apply(action)
                new_reach = [r.copy() for r in reach]
                new_reach[player] = new_reach[player] * strategy[:, a_idx]
                total_value += self._cfr(new_state, new_reach, traverser)
            return total_value

    def _terminal_value(self, state, reach, traverser):
        D = self.D
        active = state.active_players()
        t = traverser
        R = 1.0
        for j in range(self.N):
            if j != t:
                R *= np.sum(reach[j])
        if t not in active:
            return np.full(D, -state.contributions[t] * R)
        active_opponents = [p for p in active if p != t]
        if not active_opponents:
            return np.full(D, (state.pot - state.contributions[t]) * R)
        F = 1.0
        for j in range(self.N):
            if j != t and j not in active:
                F *= np.sum(reach[j])
        win_weight = np.ones(D)
        for opp in active_opponents:
            cdf = np.cumsum(reach[opp])
            cdf_shifted = np.zeros(D)
            cdf_shifted[1:] = cdf[:-1]
            win_weight *= cdf_shifted
        return state.pot * F * win_weight - state.contributions[t] * R

    # ── Projection / locking ────────────────────────────────────

    def _project_to_freq(self, strategy, reach, locks, num_actions):
        proj = strategy.copy()
        total_reach = reach.sum()
        if total_reach < 1e-15:
            return proj
        weights = reach / total_reach
        for _ in range(20):
            for a_idx, target in locks.items():
                current = (weights * proj[:, a_idx]).sum()
                if current > 1e-15:
                    proj[:, a_idx] *= target / current
                elif target > 0:
                    in_range = reach > 1e-15
                    if in_range.any():
                        proj[in_range, a_idx] = target / in_range.sum()
            proj = np.maximum(proj, 0.0)
            row_sums = proj.sum(axis=1, keepdims=True)
            proj = np.where(row_sums > 1e-15, proj / row_sums, 1.0 / num_actions)
        return proj

    def lock_node(self, player, history, locks):
        self.node_locks[(player, history)] = locks

    def unlock_node(self, player, history):
        self.node_locks.pop((player, history), None)

    def is_freq_locked(self, player, history):
        return (player, history) in self.node_locks

    def get_freq_locks(self, player, history):
        return self.node_locks.get((player, history))

    def strategy_lock_node(self, player, history, strategy):
        self.strategy_locks[(player, history)] = strategy.copy()

    def strategy_lock_current(self, player, history, num_actions):
        avg = self.get_average_strategy(player, history, num_actions)
        self.strategy_locks[(player, history)] = avg.copy()

    def strategy_unlock_node(self, player, history):
        self.strategy_locks.pop((player, history), None)

    def is_strategy_locked(self, player, history):
        return (player, history) in self.strategy_locks

    def is_locked(self, player, history):
        return self.is_freq_locked(player, history) or self.is_strategy_locked(player, history)

    def get_lock_type(self, player, history):
        if self.is_strategy_locked(player, history):
            return 'strategy'
        if self.is_freq_locked(player, history):
            return 'frequency'
        return None

    def range_lock_exploit(self, player: int, removed_intervals: List[Tuple[float, float]]):
        """Lock a player's entire strategy tree and remove hands from their range.

        1. Strategy-locks every info set for `player` to current average strategy.
        2. Zeros out `player`'s initial reach for hands in `removed_intervals`.
        3. Resets regrets for all OTHER players so they re-solve from scratch.

        After calling this, run train() to find opponents' max-EV exploit.
        """
        self._ensure_discovered()

        # Lock all of this player's info sets
        for (p, h), actions in self.info_sets.items():
            if p == player:
                avg = self.get_average_strategy(p, h, len(actions))
                self.strategy_locks[(p, h)] = avg.copy()

        # Remove hands from player's range
        for i, val in enumerate(self.hand_values):
            for lo, hi in removed_intervals:
                if lo <= val <= hi:
                    self.initial_reach[player][i] = 0.0
                    break

        # Reset opponent regrets so they re-solve fresh against the modified range
        for (p, h) in list(self._index_map.keys()):
            if p != player:
                idx = self._index_map[(p, h)]
                self.regret_data[idx] = 0.0
                self.strategy_data[idx] = 0.0

    def undo_range_lock_exploit(self):
        """Remove all strategy locks and restore original ranges."""
        self.strategy_locks.clear()

        # Restore initial reach from config
        self.initial_reach = []
        for p in range(self.N):
            reach = np.zeros(self.D)
            rc = self.config.range_configs[p]
            for i, val in enumerate(self.hand_values):
                if rc.contains(val):
                    reach[i] = 1.0 / self.D
            self.initial_reach.append(reach)

        # Reset all regrets for a clean re-solve
        if self.regret_data is not None:
            self.regret_data[:] = 0.0
            self.strategy_data[:] = 0.0

    # ── Analysis ────────────────────────────────────────────────

    def compute_reach_at_node(self, target_history):
        reach = [r.copy() for r in self.initial_reach]
        state = GameState(self.config)
        idx = 0
        while idx < len(target_history):
            if state.is_terminal():
                break
            action_key = target_history[idx]
            if action_key == '|':
                idx += 1
                continue
            player = state.current_player()
            actions = state.available_actions()
            strategy = self.get_average_strategy(player, state.history, len(actions))
            matched = False
            for a_idx, action in enumerate(actions):
                if action.key() == action_key:
                    reach[player] = reach[player] * strategy[:, a_idx]
                    state = state.apply(action)
                    matched = True
                    break
            if not matched:
                break
            idx += 1
            while idx < len(target_history) and target_history[idx] == '|':
                idx += 1
        return reach

    def compute_equity(self, history):
        if history not in {h for (_, h) in self.info_sets}:
            return None
        player = None
        for (p, h) in self.info_sets:
            if h == history:
                player = p
                break
        if player is None:
            return None
        reach = self.compute_reach_at_node(history)
        state = GameState(self.config)
        idx = 0
        while idx < len(history):
            if state.is_terminal():
                break
            action_key = history[idx]
            if action_key == '|':
                idx += 1
                continue
            actions = state.available_actions()
            for action in actions:
                if action.key() == action_key:
                    state = state.apply(action)
                    break
            idx += 1
            while idx < len(history) and history[idx] == '|':
                idx += 1
        active_opponents = [p for p in state.active_players() if p != player]
        if not active_opponents:
            return np.ones(self.D)
        equity = np.ones(self.D)
        for opp in active_opponents:
            opp_total = reach[opp].sum()
            if opp_total > 1e-15:
                cdf = np.cumsum(reach[opp] / opp_total)
                cdf_shifted = np.zeros(self.D)
                cdf_shifted[1:] = cdf[:-1]
                equity *= cdf_shifted
        return equity

    def format_history(self, history):
        if not history:
            return "(root)"
        parts = []
        street = 0
        for h in history:
            if h == '|':
                street += 1
                parts.append(f"| Street {street + 1}:")
            else:
                parts.append(h)
        return " > ".join(parts) if parts else "(root)"
