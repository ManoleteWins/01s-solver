from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class IntervalMapping:
    """Maps hands in [src_lo, src_hi] linearly to [dst_lo, dst_hi]."""
    src_lo: float
    src_hi: float
    dst_lo: float
    dst_hi: float

    def remap(self, value: float) -> float:
        if self.src_lo <= value <= self.src_hi:
            t = (value - self.src_lo) / (self.src_hi - self.src_lo) if self.src_hi > self.src_lo else 0.0
            return self.dst_lo + t * (self.dst_hi - self.dst_lo)
        return None


@dataclass
class BoardState:
    """A single possible board outcome at a street transition."""
    name: str
    weight: float = 1.0
    mappings: List[IntervalMapping] = field(default_factory=list)

    def remap_value(self, value: float) -> float:
        for m in self.mappings:
            result = m.remap(value)
            if result is not None:
                return result
        return value  # identity if not covered

    @staticmethod
    def identity(name: str = "Brick", weight: float = 1.0) -> "BoardState":
        return BoardState(name=name, weight=weight,
                          mappings=[IntervalMapping(0.0, 1.0, 0.0, 1.0)])


@dataclass
class TransitionConfig:
    """Board states for a street transition (street N -> N+1)."""
    board_states: List[BoardState] = field(default_factory=list)


@dataclass
class RangeConfig:
    """Range for a player as a list of [low, high] intervals on [0,1]."""
    intervals: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.0)])

    def contains(self, value: float) -> bool:
        return any(lo <= value <= hi for lo, hi in self.intervals)

    def __str__(self):
        return ", ".join(f"{lo}-{hi}" for lo, hi in self.intervals)

    @staticmethod
    def parse(text: str) -> "RangeConfig":
        """Parse from string like '0-0.5, 0.7-1.0'."""
        intervals = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            lo, hi = part.split("-")
            intervals.append((float(lo.strip()), float(hi.strip())))
        return RangeConfig(intervals=intervals)


@dataclass
class StreetConfig:
    """Configuration for a single street of betting."""
    bet_sizes: List[float] = field(default_factory=lambda: [1.0])
    raise_sizes: List[float] = field(default_factory=lambda: [1.0])
    max_raises: int = 1  # max number of raises allowed per street (0 = no raising)

    def __str__(self):
        return f"bets={self.bet_sizes}, raises={self.raise_sizes}"

    @staticmethod
    def parse_sizes(text: str) -> List[float]:
        """Parse comma-separated floats like '0.5, 1.0'."""
        sizes = []
        for part in text.split(","):
            part = part.strip()
            if part:
                sizes.append(float(part))
        return sizes


@dataclass
class GameConfig:
    num_players: int = 2
    num_streets: int = 1
    discretization: int = 100
    starting_pot: float = 2.0
    starting_stack: float = 100.0
    street_configs: List[StreetConfig] = field(default_factory=list)
    range_configs: List[RangeConfig] = field(default_factory=list)
    transition_configs: List[TransitionConfig] = field(default_factory=list)

    def __post_init__(self):
        while len(self.street_configs) < self.num_streets:
            self.street_configs.append(StreetConfig())
        while len(self.range_configs) < self.num_players:
            self.range_configs.append(RangeConfig())
        while len(self.transition_configs) < self.num_streets - 1:
            self.transition_configs.append(TransitionConfig())

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if self.num_players < 2:
            errors.append("Need at least 2 players")
        if self.num_streets < 1:
            errors.append("Need at least 1 street")
        if self.discretization < 10:
            errors.append("Discretization must be at least 10")
        if self.starting_pot <= 0:
            errors.append("Starting pot must be positive")
        if self.starting_stack <= 0:
            errors.append("Starting stack must be positive")
        for i, rc in enumerate(self.range_configs):
            for lo, hi in rc.intervals:
                if lo < 0 or hi > 1 or lo >= hi:
                    errors.append(f"Player {i+1} has invalid range interval [{lo}, {hi}]")
        for i, sc in enumerate(self.street_configs):
            if not sc.bet_sizes:
                errors.append(f"Street {i+1} needs at least one bet size")
        for i, tc in enumerate(self.transition_configs):
            for j, bs in enumerate(tc.board_states):
                if bs.weight <= 0:
                    errors.append(f"Transition {i+1} board '{bs.name}' must have positive weight")
                for k, m in enumerate(bs.mappings):
                    if m.src_lo >= m.src_hi:
                        errors.append(f"Transition {i+1} board '{bs.name}' mapping {k+1}: src_lo >= src_hi")
                    if m.dst_lo < 0 or m.dst_hi > 1:
                        errors.append(f"Transition {i+1} board '{bs.name}' mapping {k+1}: dest outside [0,1]")
        return errors
