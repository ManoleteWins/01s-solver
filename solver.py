from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from config import GameConfig
from game import GameState, Action


class CFRSolver:
    """Vanilla CFR solver for N-player [0,1] games."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.D = config.discretization
        self.N = config.num_players

        # Hand value midpoints for each bucket
        self.hand_values = np.linspace(0.5 / self.D, 1.0 - 0.5 / self.D, self.D)

        # Initial reach probabilities from range configs
        self.initial_reach: List[np.ndarray] = []
        for p in range(self.N):
            reach = np.zeros(self.D)
            rc = config.range_configs[p]
            for i, val in enumerate(self.hand_values):
                if rc.contains(val):
                    reach[i] = 1.0 / self.D
            self.initial_reach.append(reach)

        # Regret and strategy accumulators
        # Key: (player_index, action_history_tuple)
        # Value: np.ndarray of shape (D, num_actions)
        self.regret_sum: Dict[Tuple, np.ndarray] = {}
        self.strategy_sum: Dict[Tuple, np.ndarray] = {}

        # Collected info set keys for browsing results
        self.info_sets: Dict[Tuple, List[Action]] = {}

        # Node frequency locks: {(player, history): {action_idx: target_freq}}
        # Locked nodes must meet aggregate frequency constraints.
        # The solver still optimizes per-hand allocation within the constraint.
        self.node_locks: Dict[Tuple, Dict[int, float]] = {}

        # Strategy locks: {(player, history): np.ndarray shape (D, num_actions)}
        # Hard lock — the exact per-hand strategy is frozen. No regret updates.
        self.strategy_locks: Dict[Tuple, np.ndarray] = {}

        self.iterations_done = 0

    def _get_strategy(self, player: int, history: tuple, num_actions: int,
                       reach: Optional[np.ndarray] = None) -> np.ndarray:
        """Current strategy. Strategy lock > freq lock > regret matching."""
        key = (player, history)

        # Hard strategy lock — return exact frozen strategy
        if key in self.strategy_locks:
            return self.strategy_locks[key]

        if key not in self.regret_sum:
            self.regret_sum[key] = np.zeros((self.D, num_actions))

        regrets = np.maximum(self.regret_sum[key], 0.0)
        total = regrets.sum(axis=1, keepdims=True)
        strategy = np.full_like(regrets, 1.0 / num_actions)
        mask = (total > 0).flatten()
        if mask.any():
            strategy[mask] = regrets[mask] / total[mask]

        # Project to meet aggregate frequency constraints if freq-locked
        if key in self.node_locks and reach is not None:
            strategy = self._project_to_freq(strategy, reach, self.node_locks[key], num_actions)

        return strategy

    def get_average_strategy(self, player: int, history: tuple, num_actions: int) -> np.ndarray:
        """Average strategy over all iterations. Respects strategy/freq locks. Shape (D, num_actions)."""
        key = (player, history)

        # Hard strategy lock — return frozen strategy
        if key in self.strategy_locks:
            return self.strategy_locks[key]

        if key not in self.strategy_sum:
            avg = np.full((self.D, num_actions), 1.0 / num_actions)
        else:
            total = self.strategy_sum[key].sum(axis=1, keepdims=True)
            avg = np.where(total > 0, self.strategy_sum[key] / total, 1.0 / num_actions)

        if key in self.node_locks:
            avg = self._project_to_freq(avg, self.initial_reach[player],
                                        self.node_locks[key], num_actions)
        return avg

    def train(self, iterations: int, callback: Optional[Callable[[int, int], None]] = None):
        """Run `iterations` of vanilla CFR (alternating updates)."""
        for t in range(iterations):
            for p in range(self.N):
                root = GameState(self.config)
                reach = [r.copy() for r in self.initial_reach]
                self._cfr(root, reach, p)
            self.iterations_done += 1
            if callback:
                callback(self.iterations_done, iterations)

    def _cfr(self, state: GameState, reach: List[np.ndarray], traverser: int) -> np.ndarray:
        """
        Recursive CFR traversal for a single traverser.
        Returns counterfactual values for the traverser, shape (D,).
        """
        if state.is_terminal():
            return self._terminal_value(state, reach, traverser)

        player = state.current_player()
        actions = state.available_actions()
        num_actions = len(actions)

        # Record this info set
        info_key = (player, state.history)
        if info_key not in self.info_sets:
            self.info_sets[info_key] = actions

        strategy = self._get_strategy(player, state.history, num_actions, reach[player])

        if player == traverser:
            # Traverser's decision node
            action_values = np.zeros((num_actions, self.D))

            for a_idx, action in enumerate(actions):
                new_state = state.apply(action)
                # Pass updated reach for correct strategy_sum weighting in subtree
                new_reach = self._copy_reach(reach)
                new_reach[traverser] = new_reach[traverser] * strategy[:, a_idx]
                action_values[a_idx] = self._cfr(new_state, new_reach, traverser)

            # Node value: weighted by strategy
            node_value = np.einsum('ad,da->d', action_values, strategy)

            # Skip regret/strategy updates if strategy-locked
            key = (player, state.history)
            if key not in self.strategy_locks:
                # Regret update
                if key not in self.regret_sum:
                    self.regret_sum[key] = np.zeros((self.D, num_actions))
                for a_idx in range(num_actions):
                    self.regret_sum[key][:, a_idx] += action_values[a_idx] - node_value

                # Strategy sum weighted by player's reach to this node
                if key not in self.strategy_sum:
                    self.strategy_sum[key] = np.zeros((self.D, num_actions))
                self.strategy_sum[key] += reach[player][:, np.newaxis] * strategy

            return node_value

        else:
            # Opponent's decision node: sum over actions (reach splits)
            total_value = np.zeros(self.D)

            for a_idx, action in enumerate(actions):
                new_state = state.apply(action)
                new_reach = self._copy_reach(reach)
                new_reach[player] = new_reach[player] * strategy[:, a_idx]
                v = self._cfr(new_state, new_reach, traverser)
                total_value += v

            return total_value

    def _terminal_value(self, state: GameState, reach: List[np.ndarray], traverser: int) -> np.ndarray:
        """
        Compute counterfactual values at a terminal node for the traverser.
        Uses the CDF trick for efficient showdown computation.
        Returns shape (D,).
        """
        D = self.D
        active = state.active_players()
        t = traverser

        # Total opponent reach product: R = prod_{j!=t} sum(reach_j)
        R = 1.0
        for j in range(self.N):
            if j != t:
                R *= np.sum(reach[j])

        if t not in active:
            # Traverser folded: lost their contribution
            return np.full(D, -state.contributions[t] * R)

        active_opponents = [p for p in active if p != t]

        if not active_opponents:
            # Everyone else folded: traverser wins the pot
            return np.full(D, (state.pot - state.contributions[t]) * R)

        # Showdown with CDF trick
        # W(b) = P(traverser wins with hand bucket b) * opponent reach weighting
        # W(b) = [prod of total_reach for folded opponents] * [prod of CDF(b) for active opponents]

        # Folded opponent reach product
        F = 1.0
        for j in range(self.N):
            if j != t and j not in active:
                F *= np.sum(reach[j])

        # CDF product for active opponents
        win_weight = np.ones(D)
        for opp in active_opponents:
            cdf = np.cumsum(reach[opp])
            # Shifted: P(opponent hand < bucket b) = sum of reach[0..b-1]
            cdf_shifted = np.zeros(D)
            cdf_shifted[1:] = cdf[:-1]
            win_weight *= cdf_shifted

        # cf_v(b) = pot * F * W(b) - contribution_t * R
        return state.pot * F * win_weight - state.contributions[t] * R

    def _project_to_freq(self, strategy: np.ndarray, reach: np.ndarray,
                          locks: Dict[int, float], num_actions: int) -> np.ndarray:
        """
        Project strategy so that aggregate frequencies match locked targets.
        locks: {action_idx: target_freq} — only locked actions are constrained,
               unlocked actions absorb the slack.
        """
        proj = strategy.copy()
        total_reach = reach.sum()
        if total_reach < 1e-15:
            return proj

        weights = reach / total_reach  # (D,)

        # Iterative projection: scale locked actions, renormalize unlocked
        for _ in range(20):
            for a_idx, target in locks.items():
                current = (weights * proj[:, a_idx]).sum()
                if current > 1e-15:
                    proj[:, a_idx] *= target / current
                elif target > 0:
                    # No current mass — spread target uniformly across hands in range
                    in_range = reach > 1e-15
                    if in_range.any():
                        proj[in_range, a_idx] = target / in_range.sum()

            # Clamp and renormalize rows to valid distributions
            proj = np.maximum(proj, 0.0)
            row_sums = proj.sum(axis=1, keepdims=True)
            proj = np.where(row_sums > 1e-15, proj / row_sums, 1.0 / num_actions)

        return proj

    def lock_node(self, player: int, history: tuple, locks: Dict[int, float]):
        """Lock aggregate frequencies at a node. locks = {action_idx: target_freq}."""
        self.node_locks[(player, history)] = locks

    def unlock_node(self, player: int, history: tuple):
        """Remove frequency lock from a node."""
        self.node_locks.pop((player, history), None)

    def is_freq_locked(self, player: int, history: tuple) -> bool:
        return (player, history) in self.node_locks

    def get_freq_locks(self, player: int, history: tuple) -> Optional[Dict[int, float]]:
        return self.node_locks.get((player, history))

    def strategy_lock_node(self, player: int, history: tuple, strategy: np.ndarray):
        """Hard-lock the exact per-hand strategy at a node."""
        self.strategy_locks[(player, history)] = strategy.copy()

    def strategy_lock_current(self, player: int, history: tuple, num_actions: int):
        """Freeze the current average strategy at a node."""
        avg = self.get_average_strategy(player, history, num_actions)
        self.strategy_locks[(player, history)] = avg.copy()

    def strategy_unlock_node(self, player: int, history: tuple):
        """Remove strategy lock from a node."""
        self.strategy_locks.pop((player, history), None)

    def is_strategy_locked(self, player: int, history: tuple) -> bool:
        return (player, history) in self.strategy_locks

    def is_locked(self, player: int, history: tuple) -> bool:
        return self.is_freq_locked(player, history) or self.is_strategy_locked(player, history)

    def get_lock_type(self, player: int, history: tuple) -> Optional[str]:
        if self.is_strategy_locked(player, history):
            return 'strategy'
        if self.is_freq_locked(player, history):
            return 'frequency'
        return None

    def compute_reach_at_node(self, target_history: tuple) -> List[np.ndarray]:
        """Replay actions from root to compute reach probabilities at a given node."""
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
            num_actions = len(actions)
            strategy = self.get_average_strategy(player, state.history, num_actions)

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
            # Skip '|' separators that apply added during state transitions
            while idx < len(target_history) and target_history[idx] == '|':
                idx += 1

        return reach

    def compute_equity(self, history: tuple) -> Optional[np.ndarray]:
        """
        Compute raw showdown equity for the acting player at a node.
        Returns P(win) for each hand bucket, shape (D,), or None.
        """
        if history not in {h for (_, h) in self.info_sets}:
            return None

        # Find the player at this node
        player = None
        for (p, h) in self.info_sets:
            if h == history:
                player = p
                break
        if player is None:
            return None

        reach = self.compute_reach_at_node(history)

        # Get active players at this node by replaying state
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

        # Equity = product of CDFs of opponents' normalized reach
        equity = np.ones(self.D)
        for opp in active_opponents:
            opp_total = reach[opp].sum()
            if opp_total > 1e-15:
                cdf = np.cumsum(reach[opp] / opp_total)
                cdf_shifted = np.zeros(self.D)
                cdf_shifted[1:] = cdf[:-1]
                equity *= cdf_shifted
            # If opponent has no reach, they folded — doesn't affect equity

        return equity

    def _copy_reach(self, reach: List[np.ndarray]) -> List[np.ndarray]:
        return [r.copy() for r in reach]

    def collect_decision_points(self) -> Dict[int, List[Tuple[tuple, List[Action]]]]:
        """Group info sets by player for browsing. Returns {player: [(history, actions), ...]}."""
        by_player: Dict[int, List[Tuple[tuple, List[Action]]]] = {p: [] for p in range(self.N)}
        for (player, history), actions in sorted(self.info_sets.items()):
            by_player[player].append((history, actions))
        return by_player

    def format_history(self, history: tuple) -> str:
        """Human-readable action history."""
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
