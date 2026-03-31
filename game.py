from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from config import GameConfig


@dataclass(frozen=True)
class Action:
    type: str       # 'fold', 'check', 'call', 'bet', 'raise', 'allin'
    size: float = 0.0   # pot fraction (for bet/raise), 0 for allin
    amount: float = 0.0  # absolute chips added to pot by this action

    def key(self) -> str:
        if self.type == 'allin':
            return 'allin'
        if self.type in ('bet', 'raise'):
            return f"{self.type}:{self.size}"
        return self.type

    def label(self) -> str:
        if self.type == 'bet':
            return f"Bet {self.size:.0%} ({self.amount:.1f})"
        if self.type == 'raise':
            return f"Raise {self.size:.0%} ({self.amount:.1f})"
        if self.type == 'allin':
            return f"All-in ({self.amount:.1f})"
        if self.type == 'call':
            return f"Call ({self.amount:.1f})"
        return self.type.capitalize()


class GameState:
    __slots__ = [
        'config', 'street', 'pot', 'contributions', 'street_bets',
        'current_bet', 'active', 'all_in', 'to_act', 'history',
        '_terminal', 'num_raises_this_street', 'board_path',
    ]

    def __init__(self, config: GameConfig):
        self.config = config
        self.street = 0
        self.pot = config.starting_pot
        self.contributions = [0.0] * config.num_players
        self.street_bets = [0.0] * config.num_players
        self.current_bet = 0.0
        self.active = [True] * config.num_players
        self.all_in = [False] * config.num_players
        self.to_act = list(range(config.num_players))
        self.history: Tuple[str, ...] = ()
        self._terminal = False
        self.num_raises_this_street = 0
        self.board_path: Tuple[int, ...] = ()

    def copy(self) -> GameState:
        s = GameState.__new__(GameState)
        s.config = self.config
        s.street = self.street
        s.pot = self.pot
        s.contributions = self.contributions.copy()
        s.street_bets = self.street_bets.copy()
        s.current_bet = self.current_bet
        s.active = self.active.copy()
        s.all_in = self.all_in.copy()
        s.to_act = self.to_act.copy()
        s.history = self.history
        s._terminal = self._terminal
        s.num_raises_this_street = self.num_raises_this_street
        s.board_path = self.board_path
        return s

    def is_terminal(self) -> bool:
        return self._terminal

    def current_player(self) -> int:
        return self.to_act[0]

    def remaining_stack(self, player: int) -> float:
        return self.config.starting_stack - self.contributions[player]

    def num_active(self) -> int:
        return sum(self.active)

    def num_can_act(self) -> int:
        """Players who are active and not all-in."""
        return sum(a and not ai for a, ai in zip(self.active, self.all_in))

    def active_players(self) -> List[int]:
        return [p for p in range(self.config.num_players) if self.active[p]]

    def available_actions(self) -> List[Action]:
        player = self.current_player()
        sc = self.config.street_configs[self.street]
        remaining = self.remaining_stack(player)
        actions = []

        if self.current_bet == 0.0:
            # No bet yet: check or bet
            actions.append(Action('check'))
            for size in sc.bet_sizes:
                amount = min(size * self.pot, remaining)
                if amount > 0:
                    actions.append(Action('bet', size, amount))
            # If this is the last raise slot, also offer all-in as a bet
            # (all-in is always available as an aggressive option)
            allin_amount = remaining
            if allin_amount > 0:
                actions.append(Action('allin', 0.0, allin_amount))
        else:
            # Facing a bet: fold, call, or raise
            actions.append(Action('fold'))
            call_amount = min(self.current_bet - self.street_bets[player], remaining)
            actions.append(Action('call', amount=call_amount))

            if remaining > call_amount:
                if self.num_raises_this_street < sc.max_raises - 1:
                    # Normal raises available (not at cap yet)
                    for size in sc.raise_sizes:
                        pot_after_call = self.pot + call_amount
                        raise_extra = size * pot_after_call
                        total_amount = min(call_amount + raise_extra, remaining)
                        actions.append(Action('raise', size, total_amount))
                # At or approaching cap: all-in is the final raise
                if self.num_raises_this_street < sc.max_raises:
                    actions.append(Action('allin', 0.0, remaining))

        return actions

    def apply(self, action: Action) -> GameState:
        new = self.copy()
        player = new.to_act.pop(0)
        new.history = new.history + (action.key(),)

        if action.type == 'fold':
            new.active[player] = False

        elif action.type == 'check':
            pass

        elif action.type == 'call':
            call_amount = min(
                new.current_bet - new.street_bets[player],
                new.remaining_stack(player),
            )
            new.street_bets[player] += call_amount
            new.contributions[player] += call_amount
            new.pot += call_amount
            if new.remaining_stack(player) <= 0:
                new.all_in[player] = True

        elif action.type == 'bet':
            bet_amount = min(action.size * self.pot, self.remaining_stack(player))
            new.street_bets[player] = bet_amount
            new.contributions[player] += bet_amount
            new.pot += bet_amount
            new.current_bet = bet_amount
            if new.remaining_stack(player) <= 0:
                new.all_in[player] = True
            new.to_act = self._others_after(player, new.active, new.all_in)

        elif action.type == 'raise':
            call_amount = self.current_bet - self.street_bets[player]
            pot_after_call = self.pot + call_amount
            raise_extra = action.size * pot_after_call
            total_added = min(call_amount + raise_extra, self.remaining_stack(player))
            new.street_bets[player] += total_added
            new.contributions[player] += total_added
            new.pot += total_added
            new.current_bet = new.street_bets[player]
            new.num_raises_this_street += 1
            if new.remaining_stack(player) <= 0:
                new.all_in[player] = True
            new.to_act = self._others_after(player, new.active, new.all_in)

        elif action.type == 'allin':
            allin_amount = self.remaining_stack(player)
            new.street_bets[player] += allin_amount
            new.contributions[player] += allin_amount
            new.pot += allin_amount
            new.all_in[player] = True
            # If this raises the current bet, others must respond
            if new.street_bets[player] > new.current_bet:
                new.current_bet = new.street_bets[player]
                new.num_raises_this_street += 1
                new.to_act = self._others_after(player, new.active, new.all_in)

        # Check if betting round is over
        if not new.to_act:
            self._finish_round(new)

        return new

    def _finish_round(self, new: GameState):
        """Handle end of a betting round — advance street or go terminal."""
        n_active = new.num_active()
        n_can_act = new.num_can_act()

        # Terminal if: 1 or fewer active, last street, or everyone is all-in
        if n_active <= 1 or new.street >= new.config.num_streets - 1 or n_can_act <= 1:
            new._terminal = True
        else:
            # Advance to next street
            new.street += 1
            new.street_bets = [0.0] * new.config.num_players
            new.current_bet = 0.0
            new.num_raises_this_street = 0
            new.to_act = [
                p for p in range(new.config.num_players)
                if new.active[p] and not new.all_in[p]
            ]
            new.history = new.history + ('|',)
            # If nobody can act on next street, go terminal
            if not new.to_act:
                new._terminal = True

    def _others_after(self, player: int, active: List[bool], all_in: List[bool]) -> List[int]:
        """Active, non-all-in players in positional order after `player`."""
        n = self.config.num_players
        result = []
        for i in range(1, n):
            p = (player + i) % n
            if active[p] and not all_in[p]:
                result.append(p)
        return result
