from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import GameConfig, StreetConfig, RangeConfig
from solver import CFRSolver


class SolverGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("[0,1] Multi-Street CFR Solver")
        self.root.geometry("1200x800")

        self.solver: Optional[CFRSolver] = None
        self._running = False

        # Dynamic widget storage
        self.range_entries = []
        self.bet_entries = []
        self.raise_entries = []
        self.max_raises_entries = []

        self._build_ui()

    # ─── UI Construction ─────────────────────────────────────────

    def _build_ui(self):
        # Main horizontal panes
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: config
        left = ttk.Frame(paned, width=380)
        paned.add(left, weight=0)

        # Right: results
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._build_config_panel(left)
        self._build_results_panel(right)

    def _build_config_panel(self, parent: ttk.Frame):
        canvas = tk.Canvas(parent, width=360)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        self.config_frame = ttk.Frame(canvas)

        self.config_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.config_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        f = self.config_frame
        row = 0

        # ── Game parameters ──
        ttk.Label(f, text="Game Parameters", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 2)
        )
        row += 1

        ttk.Label(f, text="Players:").grid(row=row, column=0, sticky="w")
        self.num_players_var = tk.IntVar(value=2)
        sp = ttk.Spinbox(f, from_=2, to=6, textvariable=self.num_players_var, width=5,
                         command=self._rebuild_dynamic)
        sp.grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="Streets:").grid(row=row, column=0, sticky="w")
        self.num_streets_var = tk.IntVar(value=1)
        sp2 = ttk.Spinbox(f, from_=1, to=5, textvariable=self.num_streets_var, width=5,
                          command=self._rebuild_dynamic)
        sp2.grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="Discretization:").grid(row=row, column=0, sticky="w")
        self.disc_var = tk.IntVar(value=100)
        ttk.Spinbox(f, from_=10, to=1000, increment=10, textvariable=self.disc_var, width=7).grid(
            row=row, column=1, sticky="w"
        )
        row += 1

        ttk.Label(f, text="Starting pot:").grid(row=row, column=0, sticky="w")
        self.pot_var = tk.DoubleVar(value=2.0)
        ttk.Entry(f, textvariable=self.pot_var, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="Starting stack:").grid(row=row, column=0, sticky="w")
        self.stack_var = tk.DoubleVar(value=100.0)
        ttk.Entry(f, textvariable=self.stack_var, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Label(f, text="Iterations:").grid(row=row, column=0, sticky="w")
        self.iter_var = tk.IntVar(value=1000)
        ttk.Entry(f, textvariable=self.iter_var, width=8).grid(row=row, column=1, sticky="w")
        row += 1

        # ── Dynamic sections placeholder ──
        self.dynamic_start_row = row
        self.dynamic_frame = ttk.Frame(f)
        self.dynamic_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        # ── Run / Progress ──
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)

        self.run_btn = ttk.Button(btn_frame, text="Run Solver", command=self._run_solver)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_solver, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        row += 1

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(f, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(f, textvariable=self.status_var).grid(row=row, column=0, columnspan=2, sticky="w")

        self._rebuild_dynamic()

    def _rebuild_dynamic(self):
        """Rebuild player range and street config entries."""
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        self.range_entries.clear()
        self.bet_entries.clear()
        self.raise_entries.clear()
        self.max_raises_entries.clear()

        f = self.dynamic_frame
        row = 0

        # ── Player Ranges ──
        ttk.Label(f, text="Player Ranges", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        row += 1

        n_players = self.num_players_var.get()
        for p in range(n_players):
            ttk.Label(f, text=f"Player {p + 1}:").grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value="0-1")
            ttk.Entry(f, textvariable=var, width=25).grid(row=row, column=1, sticky="w")
            self.range_entries.append(var)
            row += 1

        # ── Street Configs ──
        ttk.Label(f, text="Street Actions", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        row += 1

        n_streets = self.num_streets_var.get()
        for s in range(n_streets):
            ttk.Label(f, text=f"Street {s + 1} bet sizes (pot frac):").grid(
                row=row, column=0, sticky="w"
            )
            bvar = tk.StringVar(value="1.0")
            ttk.Entry(f, textvariable=bvar, width=20).grid(row=row, column=1, sticky="w")
            self.bet_entries.append(bvar)
            row += 1

            ttk.Label(f, text=f"Street {s + 1} raise sizes (pot frac):").grid(
                row=row, column=0, sticky="w"
            )
            rvar = tk.StringVar(value="1.0")
            ttk.Entry(f, textvariable=rvar, width=20).grid(row=row, column=1, sticky="w")
            self.raise_entries.append(rvar)
            row += 1

            ttk.Label(f, text=f"Street {s + 1} max raises:").grid(
                row=row, column=0, sticky="w"
            )
            mrvar = tk.IntVar(value=1)
            ttk.Spinbox(f, from_=0, to=10, textvariable=mrvar, width=5).grid(
                row=row, column=1, sticky="w"
            )
            self.max_raises_entries.append(mrvar)
            row += 1

    def _build_results_panel(self, parent: ttk.Frame):
        # Breadcrumb path pills
        self.breadcrumb_frame = ttk.Frame(parent)
        self.breadcrumb_frame.pack(fill=tk.X, padx=5, pady=(5, 2))

        # Action choice pills + lock controls
        self.action_pill_frame = ttk.Frame(parent)
        self.action_pill_frame.pack(fill=tk.X, padx=5, pady=(2, 2))

        self.lock_frame = ttk.Frame(parent)
        self.lock_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Matplotlib figure: strategy + range density + freq bar
        from matplotlib.gridspec import GridSpec
        self.fig = Figure(figsize=(8, 7), dpi=100)
        gs = GridSpec(3, 1, figure=self.fig, height_ratios=[4, 4, 0.6],
                      hspace=0.35, top=0.95, bottom=0.04)
        self.ax = self.fig.add_subplot(gs[0])          # strategy (top)
        self.ax_range = self.fig.add_subplot(gs[1])     # range density (middle)
        self.ax_freq = self.fig.add_subplot(gs[2])      # frequency bar (bottom)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Navigation state: unified game tree
        self._current_history = ()
        # node_map: {history: (player, actions)} — which player acts and what actions
        self._node_map = {}
        # child_map: {(history, action_key): child_history}
        self._child_map = {}

        self._clear_plot()

    # ─── Config Parsing ──────────────────────────────────────────

    def _parse_config(self) -> Optional[GameConfig]:
        try:
            n_players = self.num_players_var.get()
            n_streets = self.num_streets_var.get()

            ranges = []
            for var in self.range_entries:
                ranges.append(RangeConfig.parse(var.get()))

            streets = []
            for i in range(n_streets):
                bets = StreetConfig.parse_sizes(self.bet_entries[i].get())
                raises = StreetConfig.parse_sizes(self.raise_entries[i].get())
                max_r = self.max_raises_entries[i].get()
                streets.append(StreetConfig(bet_sizes=bets, raise_sizes=raises, max_raises=max_r))

            cfg = GameConfig(
                num_players=n_players,
                num_streets=n_streets,
                discretization=self.disc_var.get(),
                starting_pot=self.pot_var.get(),
                starting_stack=self.stack_var.get(),
                street_configs=streets,
                range_configs=ranges,
            )

            errors = cfg.validate()
            if errors:
                messagebox.showerror("Config Error", "\n".join(errors))
                return None
            return cfg

        except Exception as e:
            messagebox.showerror("Parse Error", str(e))
            return None

    def _config_matches(self, cfg: GameConfig) -> bool:
        """Check if config matches the solver's config (so we can reuse it)."""
        if not hasattr(self, '_last_config') or self._last_config is None:
            return False
        old = self._last_config
        return (old.num_players == cfg.num_players and
                old.num_streets == cfg.num_streets and
                old.discretization == cfg.discretization and
                old.starting_pot == cfg.starting_pot and
                old.starting_stack == cfg.starting_stack)

    # ─── Solver Control ──────────────────────────────────────────

    def _run_solver(self):
        cfg = self._parse_config()
        if cfg is None:
            return

        # Reuse existing solver if config hasn't changed (preserves locks + progress)
        if self.solver is None or not self._config_matches(cfg):
            self.solver = CFRSolver(cfg)
            self._last_config = cfg
        self._running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Running...")
        self.progress_var.set(0)

        total = self.iter_var.get()

        def worker():
            try:
                self.solver.train(total, callback=self._progress_callback)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Solver Error", str(e)))
            finally:
                self.root.after(0, self._solver_done)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _stop_solver(self):
        self._running = False

    def _progress_callback(self, current: int, total: int):
        if not self._running:
            raise StopIteration("Stopped by user")
        if current % max(1, total // 100) == 0 or current == total:
            pct = 100.0 * current / total
            self.root.after(0, lambda: self.progress_var.set(pct))
            self.root.after(0, lambda c=current: self.status_var.set(f"Iteration {c}/{total}"))

    def _solver_done(self):
        self._running = False
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        locks_str = f", {len(self.solver.node_locks)} locks" if self.solver.node_locks else ""
        self.status_var.set(f"Done — {self.solver.iterations_done} iterations{locks_str}")
        self._populate_results()

    # ─── Results Display ─────────────────────────────────────────

    def _populate_results(self):
        if self.solver is None:
            return

        # Build unified node map: {history: (player, actions)}
        self._node_map = {}
        for (player, history), actions in self.solver.info_sets.items():
            self._node_map[history] = (player, actions)

        # Build child map: {(history, action_key): child_history}
        self._child_map = {}
        all_histories = sorted(self._node_map.keys(), key=len)
        for hist in all_histories:
            _, actions = self._node_map[hist]
            for action in actions:
                prefix = hist + (action.key(),)
                for child_hist in all_histories:
                    if len(child_hist) > len(hist) and child_hist[:len(prefix)] == prefix:
                        self._child_map[(hist, action.key())] = child_hist
                        break

        # Start at root (shortest history)
        self._current_history = all_histories[0] if all_histories else ()
        self._render_navigation()
        self._render_current_strategy()

    def _make_pill(self, parent, text, command=None, active=False, color=None):
        """Create a pill button."""
        bg = color or ('#4a90d9' if active else '#e0e0e0')
        fg = 'white' if active or color else '#333333'
        btn = tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=fg, relief=tk.FLAT,
            padx=10, pady=4, font=("", 9),
            activebackground=bg, activeforeground=fg,
            cursor="hand2",
            bd=0, highlightthickness=0,
        )
        btn.pack(side=tk.LEFT, padx=2, pady=2)
        return btn

    def _render_navigation(self):
        """Render breadcrumb path and action choice pills."""
        # Breadcrumbs
        for w in self.breadcrumb_frame.winfo_children():
            w.destroy()

        # Find all ancestor nodes along the current path
        ancestors = []
        all_histories = sorted(self._node_map.keys(), key=len)
        for hist in all_histories:
            if len(hist) <= len(self._current_history) and self._current_history[:len(hist)] == hist:
                ancestors.append(hist)

        for i, hist in enumerate(ancestors):
            is_current = (hist == self._current_history)
            player, _ = self._node_map[hist]

            # Build label: P{n} + what happened since last ancestor
            if i == 0:
                label = f"P{player + 1}"
            else:
                prev = ancestors[i - 1]
                segment = hist[len(prev):]
                parts = [s for s in segment if s != '|']
                has_street = '|' in segment
                action_text = " ".join(parts) if parts else ""
                prefix = f"P{player + 1}"
                if has_street:
                    street_num = sum(1 for s in hist if s == '|') + 1
                    prefix = f"St{street_num} P{player + 1}"
                label = f"{prefix}: {action_text}" if action_text else prefix

            # Mark locked nodes in breadcrumb
            node_player, _ = self._node_map[hist]
            lock_type = self.solver.get_lock_type(node_player, hist) if self.solver else None
            if lock_type == 'strategy':
                pill_label = f"[S] {label}"
                lock_color = '#8e44ad' if not is_current else None  # purple
            elif lock_type == 'frequency':
                pill_label = f"[F] {label}"
                lock_color = '#c0392b' if not is_current else None  # red
            else:
                pill_label = label
                lock_color = None

            self._make_pill(
                self.breadcrumb_frame, pill_label,
                command=lambda h=hist: self._navigate_to(h),
                active=is_current,
                color=lock_color,
            )

        # Action pills
        for w in self.action_pill_frame.winfo_children():
            w.destroy()

        if self._current_history in self._node_map:
            player, actions = self._node_map[self._current_history]
            ttk.Label(self.action_pill_frame, text=f"P{player+1} actions:",
                      font=("", 9)).pack(side=tk.LEFT, padx=(0, 5))
            for action in actions:
                has_child = (self._current_history, action.key()) in self._child_map
                color = self._action_pill_color(action)
                pill = self._make_pill(
                    self.action_pill_frame,
                    action.label() + (" >" if has_child else ""),
                    command=(lambda a=action: self._navigate_action(a)) if has_child else None,
                    color=color,
                )
                if not has_child:
                    pill.config(cursor="arrow")

        # Lock controls
        for w in self.lock_frame.winfo_children():
            w.destroy()

        if self._current_history in self._node_map and self.solver is not None:
            player, actions = self._node_map[self._current_history]
            lock_type = self.solver.get_lock_type(player, self._current_history)

            if lock_type == 'strategy':
                ttk.Label(self.lock_frame, text="STRATEGY LOCKED",
                          font=("", 9, "bold"), foreground="#8e44ad").pack(side=tk.LEFT, padx=(0, 10))
                ttk.Button(self.lock_frame, text="Edit",
                           command=self._show_strategy_editor).pack(side=tk.LEFT, padx=2)
                ttk.Button(self.lock_frame, text="Unlock",
                           command=self._unlock_current).pack(side=tk.LEFT, padx=2)
            elif lock_type == 'frequency':
                locks = self.solver.get_freq_locks(player, self._current_history)
                lock_text = ", ".join(
                    f"{actions[a].label()}={f:.0%}" for a, f in locks.items()
                )
                ttk.Label(self.lock_frame, text=f"FREQ LOCKED: {lock_text}",
                          font=("", 9, "bold"), foreground="red").pack(side=tk.LEFT, padx=(0, 10))
                ttk.Button(self.lock_frame, text="Edit",
                           command=self._show_freq_lock_dialog).pack(side=tk.LEFT, padx=2)
                ttk.Button(self.lock_frame, text="Unlock",
                           command=self._unlock_current).pack(side=tk.LEFT, padx=2)
            else:
                ttk.Button(self.lock_frame, text="Lock Frequencies",
                           command=self._show_freq_lock_dialog).pack(side=tk.LEFT, padx=2)
                ttk.Button(self.lock_frame, text="Lock Strategy",
                           command=self._show_strategy_editor).pack(side=tk.LEFT, padx=2)

    def _show_freq_lock_dialog(self):
        if self.solver is None or self._current_history not in self._node_map:
            return

        player, actions = self._node_map[self._current_history]
        history = self._current_history
        existing = self.solver.get_freq_locks(player, history) or {}

        dlg = tk.Toplevel(self.root)
        dlg.title("Lock Frequencies")
        dlg.geometry("350x300")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text=f"P{player+1} at {self.solver.format_history(history)}",
                  font=("", 10, "bold")).pack(padx=10, pady=(10, 5))
        ttk.Label(dlg, text="Set target aggregate frequency per action.\n"
                  "Leave blank to leave unlocked.\nLocked values must sum to <= 1.",
                  font=("", 8)).pack(padx=10, pady=(0, 10))

        entries = {}
        frame = ttk.Frame(dlg)
        frame.pack(padx=10, fill=tk.X)

        for a_idx, action in enumerate(actions):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=action.label(), width=15, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar()
            if a_idx in existing:
                var.set(f"{existing[a_idx]:.2f}")
            ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, padx=5)
            entries[a_idx] = var

        def apply_lock():
            locks = {}
            for a_idx, var in entries.items():
                val = var.get().strip()
                if val:
                    try:
                        f = float(val)
                        if 0 <= f <= 1:
                            locks[a_idx] = f
                        else:
                            messagebox.showerror("Error", f"Frequency must be 0-1, got {f}")
                            return
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid number: {val}")
                        return

            total = sum(locks.values())
            if total > 1.001:
                messagebox.showerror("Error", f"Locked frequencies sum to {total:.2f}, must be <= 1")
                return

            if locks:
                self.solver.lock_node(player, history, locks)
            else:
                self.solver.unlock_node(player, history)

            dlg.destroy()
            self._render_navigation()
            self._render_current_strategy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="Apply", command=apply_lock).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=5)

    def _show_strategy_editor(self):
        """Open dialog to edit per-hand strategy as range-based rules with linked sliders."""
        if self.solver is None or self._current_history not in self._node_map:
            return

        player, actions = self._node_map[self._current_history]
        history = self._current_history
        num_actions = len(actions)

        current = self.solver.get_average_strategy(player, history, num_actions)
        initial_rules = self._strategy_to_rules(current, actions)

        dlg = tk.Toplevel(self.root)
        dlg.title("Edit Strategy")
        dlg.geometry("700x550")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text=f"P{player+1} at {self.solver.format_history(history)}",
                  font=("", 10, "bold")).pack(padx=10, pady=(10, 2))
        ttk.Label(dlg, text="Define hand ranges and action probabilities. "
                  "Sliders are linked — adjusting one rebalances the others.",
                  font=("", 8), wraplength=650).pack(padx=10, pady=(0, 10))

        # Header
        header = ttk.Frame(dlg)
        header.pack(fill=tk.X, padx=10)
        ttk.Label(header, text="From", width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(header, text="To", width=6).pack(side=tk.LEFT, padx=2)
        for a in actions:
            ttk.Label(header, text=a.label(), width=12, anchor="center").pack(side=tk.LEFT, padx=2)

        # Scrollable rules area
        rules_canvas = tk.Canvas(dlg, height=320)
        rules_scroll = ttk.Scrollbar(dlg, orient=tk.VERTICAL, command=rules_canvas.yview)
        rules_frame = ttk.Frame(rules_canvas)
        rules_frame.bind("<Configure>",
                         lambda e: rules_canvas.configure(scrollregion=rules_canvas.bbox("all")))
        rules_canvas.create_window((0, 0), window=rules_frame, anchor="nw")
        rules_canvas.configure(yscrollcommand=rules_scroll.set)
        rules_canvas.pack(fill=tk.BOTH, expand=True, padx=10)
        rules_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Boundaries between rows: [0, b1, b2, ..., 1]
        # Row i covers [boundaries[i], boundaries[i+1]]
        # Only interior boundaries are editable.
        boundaries = [0.0]
        for lo, hi, _ in initial_rules:
            boundaries.append(hi)
        # Ensure last is 1
        boundaries[-1] = 1.0

        rule_rows = []       # [(boundary_var, slider_vars, row_frame), ...]
        boundary_labels = [] # ttk.Labels showing "from" values

        def rebuild_rows():
            """Rebuild all rule rows from current state."""
            for _, _, _, rf in rule_rows:
                rf.destroy()
            rule_rows.clear()
            boundary_labels.clear()

            for row_idx in range(len(boundaries) - 1):
                probs = initial_rules[row_idx][2] if row_idx < len(initial_rules) else None
                _add_row(row_idx, probs)

        def _add_row(row_idx, probs=None):
            row_frame = ttk.Frame(rules_frame)
            row_frame.pack(fill=tk.X, pady=2)

            # From label (driven by previous boundary)
            from_lbl = ttk.Label(row_frame, text=f"{boundaries[row_idx]:.2f}", width=6)
            from_lbl.pack(side=tk.LEFT, padx=2)
            boundary_labels.append(from_lbl)

            # To: editable for interior boundaries, fixed "1.00" for last row
            is_last = (row_idx == len(boundaries) - 2)
            bnd_var = tk.DoubleVar(value=boundaries[row_idx + 1])

            if is_last:
                ttk.Label(row_frame, text="1.00", width=6).pack(side=tk.LEFT, padx=2)
            else:
                bnd_entry = ttk.Entry(row_frame, textvariable=bnd_var, width=6)
                bnd_entry.pack(side=tk.LEFT, padx=2)

                def on_boundary_change(*args, idx=row_idx):
                    try:
                        val = bnd_var.get()
                    except tk.TclError:
                        return
                    # Clamp between neighbors
                    lo = boundaries[idx] + 0.01
                    hi = boundaries[idx + 2] - 0.01 if idx + 2 < len(boundaries) else 0.99
                    val = max(lo, min(hi, val))
                    boundaries[idx + 1] = round(val, 2)
                    # Update next row's from label
                    if idx + 1 < len(boundary_labels):
                        boundary_labels[idx + 1].config(text=f"{boundaries[idx + 1]:.2f}")

                bnd_var.trace_add('write', on_boundary_change)

            # Linked action sliders with per-slider pin locks
            slider_vars = []
            sliders = []
            labels_list = []
            pin_vars = []   # BooleanVar: True = pinned (won't move)
            _updating = [False]

            for a_idx in range(num_actions):
                sf = ttk.Frame(row_frame)
                sf.pack(side=tk.LEFT, padx=1)
                val = int(round((probs[a_idx] if probs else 1.0 / num_actions) * 100))
                sv = tk.IntVar(value=val)
                pv = tk.BooleanVar(value=False)

                pin_cb = ttk.Checkbutton(sf, variable=pv, text="", width=0)
                pin_cb.pack(side=tk.LEFT)
                slider = tk.Scale(sf, from_=0, to=100, orient=tk.HORIZONTAL,
                                  variable=sv, length=70, showvalue=False)
                slider.pack(side=tk.LEFT)
                lbl = ttk.Label(sf, text=f"{val}%", width=4)
                lbl.pack(side=tk.LEFT)

                slider_vars.append(sv)
                sliders.append(slider)
                labels_list.append(lbl)
                pin_vars.append(pv)

            def on_slider_change(changed_idx, *args):
                if _updating[0]:
                    return
                _updating[0] = True

                new_val = slider_vars[changed_idx].get()
                # "others" = unlocked sliders that aren't the one being moved
                others = [i for i in range(num_actions)
                          if i != changed_idx and not pin_vars[i].get()]
                pinned_sum = sum(slider_vars[i].get() for i in range(num_actions)
                                if i != changed_idx and pin_vars[i].get())
                remaining = 100 - new_val - pinned_sum
                remaining = max(0, remaining)

                old_others_sum = sum(slider_vars[i].get() for i in others)

                if old_others_sum > 0 and others:
                    for i in others:
                        scaled = int(round(slider_vars[i].get() * remaining / old_others_sum))
                        slider_vars[i].set(max(0, min(100, scaled)))
                elif remaining > 0 and others:
                    each = remaining // len(others)
                    for i in others:
                        slider_vars[i].set(each)
                    diff = remaining - each * len(others)
                    if diff > 0:
                        slider_vars[others[0]].set(slider_vars[others[0]].get() + diff)
                elif others:
                    for i in others:
                        slider_vars[i].set(0)

                # Fix rounding among unlocked
                total = sum(sv.get() for sv in slider_vars)
                if total != 100 and others:
                    slider_vars[others[-1]].set(slider_vars[others[-1]].get() + (100 - total))

                for i in range(num_actions):
                    labels_list[i].config(text=f"{slider_vars[i].get()}%")

                _updating[0] = False

            for a_idx in range(num_actions):
                sliders[a_idx].config(
                    command=lambda val, idx=a_idx: on_slider_change(idx)
                )

            # Split button: split this row into two at midpoint
            def split_row(idx=row_idx):
                mid = round((boundaries[idx] + boundaries[idx + 1]) / 2, 2)
                boundaries.insert(idx + 1, mid)
                # Duplicate current probs for the new rule
                cur_probs = [sv.get() / 100.0 for sv in slider_vars]
                initial_rules.insert(idx + 1, (mid, boundaries[idx + 2], cur_probs))
                if idx < len(initial_rules):
                    initial_rules[idx] = (boundaries[idx], mid, cur_probs)
                rebuild_rows()

            ttk.Button(row_frame, text="/", width=2, command=split_row).pack(side=tk.LEFT, padx=2)

            # Remove button (only if more than 1 row)
            def remove_row(idx=row_idx):
                if len(boundaries) <= 2:
                    return
                del boundaries[idx + 1]
                if idx < len(initial_rules):
                    initial_rules.pop(idx)
                rebuild_rows()

            ttk.Button(row_frame, text="X", width=2, command=remove_row).pack(side=tk.LEFT, padx=2)

            entry = (bnd_var, slider_vars, row_frame)
            rule_rows.append(entry)

        rebuild_rows()

        def apply_strategy():
            D = self.solver.D
            hand_values = self.solver.hand_values
            strategy = np.full((D, num_actions), 1.0 / num_actions)

            rules = []
            for row_idx, (_, slider_vars, _) in enumerate(rule_rows):
                lo = boundaries[row_idx]
                hi = boundaries[row_idx + 1]
                probs = [sv.get() / 100.0 for sv in slider_vars]
                rules.append((lo, hi, probs))

            for i, val in enumerate(hand_values):
                for lo, hi, probs in rules:
                    if lo <= val <= hi:
                        strategy[i] = probs
                        break

            self.solver.strategy_lock_node(player, history, strategy)
            dlg.destroy()
            self._render_navigation()
            self._render_current_strategy()

        btn_frame2 = ttk.Frame(dlg)
        btn_frame2.pack(pady=10)
        ttk.Button(btn_frame2, text="Apply & Lock", command=apply_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame2, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=5)

    def _strategy_to_rules(self, strategy, actions, threshold=0.05):
        """Convert a (D, num_actions) strategy into threshold-based rules."""
        D = strategy.shape[0]
        hand_values = self.solver.hand_values
        rules = []

        # Group consecutive hands with similar strategies
        current_start = 0
        current_probs = strategy[0]

        for i in range(1, D):
            if np.max(np.abs(strategy[i] - current_probs)) > threshold:
                # New segment
                lo = hand_values[current_start] - 0.5 / D
                hi = hand_values[i - 1] + 0.5 / D
                rules.append((max(0, lo), min(1, hi), current_probs.tolist()))
                current_start = i
                current_probs = strategy[i]

        # Final segment
        lo = hand_values[current_start] - 0.5 / D
        hi = hand_values[D - 1] + 0.5 / D
        rules.append((max(0, lo), min(1, hi), current_probs.tolist()))

        # Merge very small segments and limit total rules
        if len(rules) > 10:
            # Simplify: just use ~5 even segments
            rules = []
            n_seg = 5
            seg_size = D // n_seg
            for s in range(n_seg):
                start = s * seg_size
                end = min((s + 1) * seg_size, D)
                avg_probs = strategy[start:end].mean(axis=0).tolist()
                lo = hand_values[start] - 0.5 / D
                hi = hand_values[end - 1] + 0.5 / D
                rules.append((max(0, lo), min(1, hi), avg_probs))

        return rules

    def _unlock_current(self):
        if self.solver is None or self._current_history not in self._node_map:
            return
        player, _ = self._node_map[self._current_history]
        self.solver.unlock_node(player, self._current_history)
        self.solver.strategy_unlock_node(player, self._current_history)
        self._render_navigation()
        self._render_current_strategy()

    def _action_pill_color(self, action):
        if action.type in ('check', 'call'):
            return '#27ae60'
        elif action.type == 'fold':
            return '#2980b9'
        elif action.type == 'allin':
            return '#4A2000'
        elif action.type in ('bet', 'raise'):
            t = min(action.size / 2.0, 1.0)
            r = int(0xCD + (0x5A - 0xCD) * t)
            g = int(0x85 + (0x2D - 0x85) * t)
            b = int(0x3F + (0x00 - 0x3F) * t)
            return f'#{r:02X}{g:02X}{b:02X}'
        return '#888888'

    def _navigate_to(self, history):
        self._current_history = history
        self._render_navigation()
        self._render_current_strategy()

    def _navigate_action(self, action):
        key = (self._current_history, action.key())
        if key in self._child_map:
            self._current_history = self._child_map[key]
            self._render_navigation()
            self._render_current_strategy()

    def _render_current_strategy(self):
        if self._current_history not in self._node_map:
            self._clear_plot()
            return
        player, actions = self._node_map[self._current_history]
        avg_strategy = self.solver.get_average_strategy(player, self._current_history, len(actions))
        self._plot_strategy(player, self._current_history, actions, avg_strategy)

    def _plot_strategy(self, player: int, history: tuple, actions, strategy: np.ndarray):
        """Stacked area chart of action probabilities vs hand value."""
        self.ax.clear()

        x = self.solver.hand_values
        labels = [a.label() for a in actions]

        # Color mapping: check/call = green, fold = blue, bet/raise/allin = browns
        # Darker brown = bigger size, all-in = darkest
        brown_light = '#DEB887'  # lightest
        brown_dark = '#4A2000'   # darkest

        # Collect all aggressive actions and sort by size for color assignment
        aggressive = [(i, a) for i, a in enumerate(actions) if a.type in ('bet', 'raise', 'allin')]
        # Sort by size (all-in gets max size so it's always darkest)
        max_size = max((a.size for _, a in aggressive if a.type != 'allin'), default=1.0)
        def sort_key(pair):
            _, a = pair
            return max_size + 1 if a.type == 'allin' else a.size
        aggressive.sort(key=sort_key)

        # Assign browns: interpolate from light to dark
        agg_colors = {}
        n_agg = len(aggressive)
        for rank, (idx, a) in enumerate(aggressive):
            t = rank / max(n_agg - 1, 1)  # 0=lightest, 1=darkest
            r = int(0xDE + (0x4A - 0xDE) * t)
            g = int(0xB8 + (0x20 - 0xB8) * t)
            b = int(0x87 + (0x00 - 0x87) * t)
            agg_colors[idx] = f'#{r:02X}{g:02X}{b:02X}'

        colors = []
        for i, a in enumerate(actions):
            if a.type in ('check', 'call'):
                colors.append('#2ecc71')
            elif a.type == 'fold':
                colors.append('#3498db')
            elif i in agg_colors:
                colors.append(agg_colors[i])
            else:
                colors.append('#9b59b6')

        # Stacked area
        bottom = np.zeros(self.solver.D)
        for a_idx in range(len(actions)):
            probs = strategy[:, a_idx]
            self.ax.fill_between(x, bottom, bottom + probs,
                                 alpha=0.7, color=colors[a_idx],
                                 label=labels[a_idx])
            bottom = bottom + probs

        # Equity overlay
        equity = self.solver.compute_equity(history)
        if equity is not None:
            self.ax.plot(x, equity, color='black', linewidth=2, linestyle='--',
                         label='Equity', zorder=10)

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("")
        self.ax.set_ylabel("Action Probability / Equity")
        title = f"Player {player + 1} — {self.solver.format_history(history)}"
        self.ax.set_title(title, fontsize=10)
        self.ax.legend(loc='upper left', fontsize=9)
        self.ax.grid(True, alpha=0.3)

        # Aggregate frequency bar
        self.ax_freq.clear()
        reach = self.solver.compute_reach_at_node(history)
        player_reach = reach[player]
        total_reach = player_reach.sum()
        if total_reach > 0:
            weights = player_reach / total_reach
        else:
            weights = np.ones(self.solver.D) / self.solver.D

        freq_left = 0.0
        for a_idx in range(len(actions)):
            freq = (weights * strategy[:, a_idx]).sum()
            self.ax_freq.barh(0, freq, left=freq_left, height=0.6,
                              color=colors[a_idx], alpha=0.85)
            if freq > 0.04:
                self.ax_freq.text(freq_left + freq / 2, 0, f"{freq:.0%}",
                                  ha='center', va='center', fontsize=8, fontweight='bold',
                                  color='white')
            freq_left += freq
        self.ax_freq.set_xlim(0, 1)
        self.ax_freq.set_ylim(-0.4, 0.4)
        self.ax_freq.set_yticks([])
        self.ax_freq.set_xticks([])
        self.ax_freq.set_title("Aggregate Frequencies", fontsize=9, pad=2)
        for spine in self.ax_freq.spines.values():
            spine.set_visible(False)

        # Range density plot
        self.ax_range.clear()
        density = weights

        # Stacked by action: show how the range splits across actions
        bottom = np.zeros(self.solver.D)
        for a_idx in range(len(actions)):
            action_density = density * strategy[:, a_idx]
            self.ax_range.fill_between(x, bottom, bottom + action_density,
                                        alpha=0.7, color=colors[a_idx])
            bottom = bottom + action_density

        self.ax_range.set_xlim(0, 1)
        self.ax_range.set_xlabel("Hand Value")
        self.ax_range.set_ylabel("Range Density")
        self.ax_range.set_title("Range (weighted by reach)", fontsize=9)
        self.ax_range.grid(True, alpha=0.3)

        self.canvas.draw()

    def _clear_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Action Probability")
        self.ax.set_title("Run solver to see results")
        self.ax.grid(True, alpha=0.3)

        self.ax_freq.clear()
        self.ax_freq.set_xlim(0, 1)
        self.ax_freq.set_yticks([])
        self.ax_freq.set_xticks([])
        for spine in self.ax_freq.spines.values():
            spine.set_visible(False)

        self.ax_range.clear()
        self.ax_range.set_xlim(0, 1)
        self.ax_range.set_xlabel("Hand Value")
        self.ax_range.set_ylabel("Range Density")
        self.ax_range.grid(True, alpha=0.3)

        self.canvas.draw()
