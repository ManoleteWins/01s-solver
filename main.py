#!/usr/bin/env python3
"""Multi-street [0,1] game solver with vanilla CFR and GUI."""

import tkinter as tk
from gui import SolverGUI


def main():
    root = tk.Tk()
    app = SolverGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
