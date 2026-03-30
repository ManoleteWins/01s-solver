# [0,1] Multi-Street CFR Solver

A vanilla CFR (Counterfactual Regret Minimization) solver for N-player continuous [0,1] poker games with a GUI.

Players are dealt hands from the [0,1] interval. No community cards. Highest hand wins at showdown.

## Install & Run

```bash
pip install numpy matplotlib
python main.py
```

Requires Python 3.8+.

## Features

- **N players** (2-6), **N streets**, configurable discretization (default 100)
- **Pot-relative bet/raise sizing** with configurable max raises per street (final raise = all-in)
- **Ranges with gaps** (e.g. `0-0.3, 0.6-1.0`)
- **Shared-memory parallel CFR** (PioSOLVER-style) -- configurable threads
- **Node locking**:
  - **Frequency lock**: lock aggregate action frequencies, solver optimizes per-hand allocation
  - **Strategy lock**: freeze exact per-hand strategy with visual editor (linked sliders)
  - **Range exploit**: lock a player's full tree, remove hands, opponents find max-EV response
- **Visualization**:
  - Stacked strategy chart (action probabilities vs hand value)
  - Equity overlay (showdown win probability)
  - Range density chart (reach-weighted hand distribution per action)
  - Aggregate frequency bar
  - Hover tooltips with per-hand details
- **Pill-based tree navigation** through the game tree

## Configuration

| Field | Description | Default |
|-------|-------------|---------|
| Players | Number of players | 2 |
| Streets | Number of betting streets | 1 |
| Discretization | Hand value buckets | 100 |
| Starting pot | Initial pot size | 10 |
| Starting stack | Each player's stack | 100 |
| Iterations | CFR iterations to run | 10000 |
| Threads | Parallel workers | 32 |
| Bet sizes | Pot fractions, comma-separated (e.g. `0.5, 1.0`) | `1.0` |
| Raise sizes | Pot fractions for raises | `1.0` |
| Max raises | Raises per street (final = all-in) | 2 |
| Ranges | Per-player hand intervals (e.g. `0-0.5, 0.7-1.0`) | `0-1` |
