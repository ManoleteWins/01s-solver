from __future__ import annotations
import numpy as np
from multiprocessing import Process, Value, shared_memory
from typing import Dict, List, Tuple, Callable, Optional
from config import GameConfig
from game import GameState, Action


def _recompute_boards_worker(config, hand_values, D):
    """Recompute board precomputation in a worker process."""
    from itertools import product as iterproduct
    precomputed = {}
    # Enumerate all board paths
    transition_counts = []
    for tc in config.transition_configs:
        if tc.board_states:
            transition_counts.append(len(tc.board_states))
    if not transition_counts:
        return precomputed
    for path in iterproduct(*(range(n) for n in transition_counts)):
        effective = hand_values.copy()
        for step, board_idx in enumerate(path):
            tc = config.transition_configs[step]
            bs = tc.board_states[board_idx]
            effective = np.array([bs.remap_value(v) for v in effective])
        sort_order = np.argsort(effective, kind='stable')
        inv_order = np.empty_like(sort_order)
        inv_order[sort_order] = np.arange(D)
        precomputed[path] = (sort_order, inv_order)
    return precomputed


def _shared_worker(config, initial_reach, node_locks, strategy_locks,
                   index_map, n_actions_map, D, max_actions,
                   regret_shm_name, strategy_shm_name, arr_shape,
                   iterations, progress_counter,
                   locked_shm_name=None, locked_keys=None):
    """Worker: run CFR iterations reading/writing shared memory arrays."""
    regret_shm = shared_memory.SharedMemory(name=regret_shm_name)
    strategy_shm = shared_memory.SharedMemory(name=strategy_shm_name)
    regret_arr = np.ndarray(arr_shape, dtype=np.float64, buffer=regret_shm.buf)
    strategy_arr = np.ndarray(arr_shape, dtype=np.float64, buffer=strategy_shm.buf)

    locked_shm = None
    if locked_shm_name and locked_keys:
        locked_shm = shared_memory.SharedMemory(name=locked_shm_name)
        locked_arr = np.ndarray(arr_shape, dtype=np.float64, buffer=locked_shm.buf)
        strategy_locks = {}
        for key in locked_keys:
            idx = index_map[key]
            na = n_actions_map[key]
            strategy_locks[key] = locked_arr[idx, :, :na]

    N = config.num_players
    hand_values = np.linspace(0.5 / D, 1.0 - 0.5 / D, D)
    board_precomputed = _recompute_boards_worker(config, hand_values, D)
    transition_configs = config.transition_configs

    for _ in range(iterations):
        for p in range(N):
            root = GameState(config)
            reach = [r.copy() for r in initial_reach]
            _cfr_shared(root, reach, p, D, N, regret_arr, strategy_arr,
                        index_map, n_actions_map, node_locks, strategy_locks,
                        initial_reach, transition_configs, board_precomputed)
        with progress_counter.get_lock():
            progress_counter.value += 1

    regret_shm.close()
    strategy_shm.close()
    if locked_shm:
        locked_shm.close()


def _cfr_shared_recurse(state, new_state, reach, traverser, D, N,
                        regret_arr, strategy_arr, index_map, n_actions_map,
                        node_locks, strategy_locks, initial_reach,
                        transition_configs, board_precomputed):
    """Recurse into new_state, inserting chance nodes at street transitions."""
    if new_state.street > state.street:
        street = state.street
        if street < len(transition_configs):
            tc = transition_configs[street]
            if tc.board_states:
                total_weight = sum(bs.weight for bs in tc.board_states)
                value = np.zeros(D)
                for k, bs in enumerate(tc.board_states):
                    prob = bs.weight / total_weight
                    board_state = new_state.copy()
                    board_state.history = new_state.history + (f'board:{k}',)
                    board_state.board_path = state.board_path + (k,)
                    value += prob * _cfr_shared(
                        board_state, reach, traverser, D, N,
                        regret_arr, strategy_arr, index_map, n_actions_map,
                        node_locks, strategy_locks, initial_reach,
                        transition_configs, board_precomputed)
                return value
    return _cfr_shared(
        new_state, reach, traverser, D, N,
        regret_arr, strategy_arr, index_map, n_actions_map,
        node_locks, strategy_locks, initial_reach,
        transition_configs, board_precomputed)


def _cfr_shared(state, reach, traverser, D, N, regret_arr, strategy_arr,
                index_map, n_actions_map, node_locks, strategy_locks,
                initial_reach, transition_configs=None, board_precomputed=None):
    """CFR traversal using shared memory arrays."""
    if state.is_terminal():
        return _terminal_shared(state, reach, traverser, D, N, board_precomputed)

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
            action_values[a_idx] = _cfr_shared_recurse(
                state, new_state, new_reach, traverser, D, N,
                regret_arr, strategy_arr, index_map, n_actions_map,
                node_locks, strategy_locks, initial_reach,
                transition_configs, board_precomputed)

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
            total_value += _cfr_shared_recurse(
                state, new_state, new_reach, traverser, D, N,
                regret_arr, strategy_arr, index_map, n_actions_map,
                node_locks, strategy_locks, initial_reach,
                transition_configs, board_precomputed)
        return total_value


def _terminal_shared(state, reach, traverser, D, N, board_precomputed=None):
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
    if board_precomputed and state.board_path and state.board_path in board_precomputed:
        sort_order, inv_order = board_precomputed[state.board_path]
        for opp in active_opponents:
            opp_sorted = reach[opp][sort_order]
            cdf = np.cumsum(opp_sorted)
            cdf_shifted = np.zeros(D)
            cdf_shifted[1:] = cdf[:-1]
            win_weight *= cdf_shifted[inv_order]
    else:
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

        # Board state precomputation: {board_path: (sort_order, inv_order)}
        self._board_precomputed: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._has_boards = any(
            tc.board_states for tc in config.transition_configs
        )

    # ── Board precomputation ───────────────────────────────────

    def _precompute_boards(self):
        """Precompute sort orders for all board paths found during discovery."""
        if not self._has_boards:
            return
        for board_path in self._board_paths:
            if board_path in self._board_precomputed:
                continue
            effective = self.hand_values.copy()
            for step, board_idx in enumerate(board_path):
                tc = self.config.transition_configs[step]
                bs = tc.board_states[board_idx]
                effective = np.array([bs.remap_value(v) for v in effective])
            sort_order = np.argsort(effective, kind='stable')
            inv_order = np.empty_like(sort_order)
            inv_order[sort_order] = np.arange(self.D)
            self._board_precomputed[board_path] = (sort_order, inv_order)

    def _get_transition(self, street: int):
        """Get the transition config for moving from `street` to `street+1`."""
        if street < len(self.config.transition_configs):
            tc = self.config.transition_configs[street]
            if tc.board_states:
                return tc
        return None

    # ── Tree discovery ──────────────────────────────────────────

    def _discover_tree(self):
        """Traverse game tree once to find all info sets."""
        self.info_sets = {}
        self._board_paths = set()
        self._board_paths.add(())
        self._discover(GameState(self.config))

        self._precompute_boards()

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
                new_state = state.apply(action)
                if new_state.street > state.street:
                    transition = self._get_transition(state.street)
                    if transition:
                        for k in range(len(transition.board_states)):
                            bs = new_state.copy()
                            bs.history = new_state.history + (f'board:{k}',)
                            bs.board_path = state.board_path + (k,)
                            self._board_paths.add(bs.board_path)
                            self._discover(bs)
                    else:
                        self._discover(new_state)
                else:
                    self._discover(new_state)

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

        # Put strategy locks in shared memory to avoid pickling numpy arrays
        locked_shm = None
        locked_shm_name = None
        locked_keys = None
        locks_to_pass = self.strategy_locks
        if self.strategy_locks:
            locked_shm = shared_memory.SharedMemory(create=True, size=nbytes)
            locked_arr = np.ndarray(shape, dtype=np.float64, buffer=locked_shm.buf)
            locked_arr[:] = 0.0
            for key, strat in self.strategy_locks.items():
                idx = self._index_map[key]
                na = self._n_actions_map[key]
                locked_arr[idx, :, :na] = strat
            locked_shm_name = locked_shm.name
            locked_keys = frozenset(self.strategy_locks.keys())
            locks_to_pass = {}

        # Spawn workers
        processes = []
        for n in per_worker:
            if n <= 0:
                continue
            p = Process(target=_shared_worker, args=(
                self.config, self.initial_reach, self.node_locks, locks_to_pass,
                self._index_map, self._n_actions_map, self.D, self._max_actions,
                regret_shm.name, strategy_shm.name, shape, n, progress,
                locked_shm_name, locked_keys))
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
        if locked_shm:
            locked_shm.close()
            locked_shm.unlink()

    # ── CFR traversal (single-thread, uses self arrays) ─────────

    def _cfr_recurse(self, state: GameState, new_state: GameState,
                     reach: List[np.ndarray], traverser: int) -> np.ndarray:
        """Recurse into new_state, inserting chance nodes at street transitions."""
        if new_state.street > state.street:
            transition = self._get_transition(state.street)
            if transition:
                total_weight = sum(bs.weight for bs in transition.board_states)
                value = np.zeros(self.D)
                for k, bs in enumerate(transition.board_states):
                    prob = bs.weight / total_weight
                    board_state = new_state.copy()
                    board_state.history = new_state.history + (f'board:{k}',)
                    board_state.board_path = state.board_path + (k,)
                    value += prob * self._cfr(board_state, reach, traverser)
                return value
        return self._cfr(new_state, reach, traverser)

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
                action_values[a_idx] = self._cfr_recurse(state, new_state, new_reach, traverser)

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
                total_value += self._cfr_recurse(state, new_state, new_reach, traverser)
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
        if state.board_path and state.board_path in self._board_precomputed:
            sort_order, inv_order = self._board_precomputed[state.board_path]
            for opp in active_opponents:
                opp_sorted = reach[opp][sort_order]
                cdf = np.cumsum(opp_sorted)
                cdf_shifted = np.zeros(D)
                cdf_shifted[1:] = cdf[:-1]
                # Map back to original hand indices
                win_weight *= cdf_shifted[inv_order]
        else:
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

    def range_lock_exploit(self, player: int, modifications):
        """Lock a player's entire strategy tree and modify their range.

        modifications: list of (lo, hi, keep_fraction) tuples.
            keep_fraction=0 removes entirely, 0.5 keeps half, 1.0 keeps all.

        1. Strategy-locks every info set for `player` to current average strategy.
        2. Scales `player`'s initial reach by keep_fraction for hands in each interval.
        3. Resets regrets for all OTHER players so they re-solve from scratch.

        After calling this, run train() to find opponents' max-EV exploit.
        """
        self._ensure_discovered()

        # Lock all of this player's info sets
        for (p, h), actions in self.info_sets.items():
            if p == player:
                avg = self.get_average_strategy(p, h, len(actions))
                self.strategy_locks[(p, h)] = avg.copy()

        # Modify hands in player's range
        for i, val in enumerate(self.hand_values):
            for lo, hi, keep in modifications:
                if lo <= val <= hi:
                    self.initial_reach[player][i] *= keep
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

    def average_regret_pct(self):
        """Average positive regret as % of pot."""
        if self.regret_data is None or self.iterations_done == 0:
            return float('inf')
        pos_regret = np.maximum(self.regret_data, 0.0).sum()
        normalized = pos_regret / (self._n_info_sets * self.D * self.iterations_done)
        return normalized / self.config.starting_pot * 100.0

    # ── Analysis ────────────────────────────────────────────────

    def _replay_state(self, history):
        """Replay actions to reconstruct GameState at a given history."""
        state = GameState(self.config)
        idx = 0
        while idx < len(history):
            if state.is_terminal():
                break
            action_key = history[idx]
            if action_key == '|':
                idx += 1
                continue
            if action_key.startswith('board:'):
                board_idx = int(action_key.split(':')[1])
                state.board_path = state.board_path + (board_idx,)
                state.history = state.history + (action_key,)
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
        return state

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
            if action_key.startswith('board:'):
                board_idx = int(action_key.split(':')[1])
                state.board_path = state.board_path + (board_idx,)
                state.history = state.history + (action_key,)
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

    def _get_board_path_from_history(self, history):
        """Extract the board path tuple from a history."""
        path = []
        for h in history:
            if isinstance(h, str) and h.startswith('board:'):
                path.append(int(h.split(':')[1]))
        return tuple(path)

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
        state = self._replay_state(history)
        active_opponents = [p for p in state.active_players() if p != player]
        if not active_opponents:
            return np.ones(self.D)
        equity = np.ones(self.D)
        board_path = self._get_board_path_from_history(history)
        if board_path and board_path in self._board_precomputed:
            sort_order, inv_order = self._board_precomputed[board_path]
            for opp in active_opponents:
                opp_total = reach[opp].sum()
                if opp_total > 1e-15:
                    opp_sorted = reach[opp][sort_order] / opp_total
                    cdf = np.cumsum(opp_sorted)
                    cdf_shifted = np.zeros(self.D)
                    cdf_shifted[1:] = cdf[:-1]
                    equity *= cdf_shifted[inv_order]
        else:
            for opp in active_opponents:
                opp_total = reach[opp].sum()
                if opp_total > 1e-15:
                    cdf = np.cumsum(reach[opp] / opp_total)
                    cdf_shifted = np.zeros(self.D)
                    cdf_shifted[1:] = cdf[:-1]
                    equity *= cdf_shifted
        return equity

    def compute_ev(self, history):
        """Compute per-hand EV for the acting player at this node.

        Returns (ev_array, pot) where ev_array[i] is the EV in chips for hand i,
        and pot is the pot size at this node.  Returns (None, None) if the node
        is not found.
        """
        if history not in {h for (_, h) in self.info_sets}:
            return None, None
        player = None
        for (p, h) in self.info_sets:
            if h == history:
                player = p
                break
        if player is None:
            return None, None

        reach = self.compute_reach_at_node(history)
        state = self._replay_state(history)
        pot = state.pot

        # Tree walk for counterfactual values
        cf_values = self._ev_walk(state, reach, player)

        # Normalise by opponent reach product to get per-hand EV in chips
        opp_product = 1.0
        for p in range(self.N):
            if p != player:
                opp_product *= reach[p].sum()

        if opp_product > 1e-15:
            ev = cf_values / opp_product
        else:
            ev = np.zeros(self.D)

        return ev, pot

    def _ev_walk_recurse(self, state, new_state, reach, traverser):
        """Recurse into new_state for EV walk, handling chance nodes."""
        if new_state.street > state.street:
            transition = self._get_transition(state.street)
            if transition:
                total_weight = sum(bs.weight for bs in transition.board_states)
                value = np.zeros(self.D)
                for k, bs in enumerate(transition.board_states):
                    prob = bs.weight / total_weight
                    board_state = new_state.copy()
                    board_state.history = new_state.history + (f'board:{k}',)
                    board_state.board_path = state.board_path + (k,)
                    value += prob * self._ev_walk(board_state, reach, traverser)
                return value
        return self._ev_walk(new_state, reach, traverser)

    def _ev_walk(self, state, reach, traverser):
        """Recursive tree walk computing counterfactual values using average strategies."""
        if state.is_terminal():
            return self._terminal_value(state, reach, traverser)

        player = state.current_player()
        actions = state.available_actions()
        num_actions = len(actions)
        strategy = self.get_average_strategy(player, state.history, num_actions)

        if player == traverser:
            action_values = np.zeros((num_actions, self.D))
            for a_idx, action in enumerate(actions):
                new_state = state.apply(action)
                new_reach = [r.copy() for r in reach]
                new_reach[traverser] = new_reach[traverser] * strategy[:, a_idx]
                action_values[a_idx] = self._ev_walk_recurse(state, new_state, new_reach, traverser)
            return np.einsum('ad,da->d', action_values, strategy)
        else:
            total_value = np.zeros(self.D)
            for a_idx, action in enumerate(actions):
                new_state = state.apply(action)
                new_reach = [r.copy() for r in reach]
                new_reach[player] = new_reach[player] * strategy[:, a_idx]
                total_value += self._ev_walk_recurse(state, new_state, new_reach, traverser)
            return total_value

    def format_history(self, history):
        if not history:
            return "(root)"
        parts = []
        street = 0
        for h in history:
            if h == '|':
                street += 1
                parts.append(f"| Street {street + 1}:")
            elif h.startswith('board:'):
                board_idx = int(h.split(':')[1])
                if street - 1 < len(self.config.transition_configs):
                    tc = self.config.transition_configs[street - 1]
                    if board_idx < len(tc.board_states):
                        parts.append(f"[{tc.board_states[board_idx].name}]")
                    else:
                        parts.append(h)
                else:
                    parts.append(h)
            else:
                parts.append(h)
        return " > ".join(parts) if parts else "(root)"
