from dataclasses import dataclass, field
from typing import List, Tuple


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

    def __post_init__(self):
        while len(self.street_configs) < self.num_streets:
            self.street_configs.append(StreetConfig())
        while len(self.range_configs) < self.num_players:
            self.range_configs.append(RangeConfig())

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
        return errors
