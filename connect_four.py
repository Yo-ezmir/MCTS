"""
connect_four.py
═══════════════
The game engine for Connect Four.
Completely decoupled from any AI agent — knows only the rules.

Board layout
------------
  self.grid[row][col]
  row 0 = TOP, row 5 = BOTTOM  (gravity fills from bottom)
  col 0 = left, col 6 = right

Cell values
-----------
  0 = empty   1 = Player 1 (Red)   2 = Player 2 (Yellow)
"""

import numpy as np
from typing import Optional, List, Tuple


ROWS    = 6
COLS    = 7
EMPTY   = 0
P1      = 1   # Red
P2      = 2   # Yellow


class ConnectFourBoard:
    """
    Immutable-style board — always clone before mutating in the MCTS tree.
    """

    def __init__(self):
        self.grid        = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current     = P1          # whose turn it is
        self.last_move   = None        # (row, col) of the most recent drop
        self.move_count  = 0
        self._winner     = None        # cached winner: None / P1 / P2
        self._is_draw    = False

    # ------------------------------------------------------------------ #
    #  Core game operations                                                #
    # ------------------------------------------------------------------ #

    def clone(self) -> "ConnectFourBoard":
        """Deep copy — essential for MCTS tree expansion."""
        b = ConnectFourBoard()
        b.grid       = self.grid.copy()
        b.current    = self.current
        b.last_move  = self.last_move
        b.move_count = self.move_count
        b._winner    = self._winner
        b._is_draw   = self._is_draw
        return b

    def legal_moves(self) -> List[int]:
        """Return list of columns that are not full."""
        return [c for c in range(COLS) if self.grid[0][c] == EMPTY]

    def drop(self, col: int) -> "ConnectFourBoard":
        """
        Drop the current player's piece into `col`.
        Returns self (mutates in place) — clone first if you want immutability.

        Gravity: piece falls to the lowest empty row in the column.
        """
        # Find the lowest empty row in this column (gravity)
        row = None
        for r in range(ROWS - 1, -1, -1):     # scan from bottom upward
            if self.grid[r][col] == EMPTY:
                row = r
                break
        if row is None:
            raise ValueError(f"Column {col} is full.")

        self.grid[row][col] = self.current
        self.last_move       = (row, col)
        self.move_count     += 1

        # Check terminal conditions after this move
        if self._check_win(row, col):
            self._winner = self.current
        elif self.move_count == ROWS * COLS:
            self._is_draw = True

        # Switch turn
        self.current = P2 if self.current == P1 else P1
        return self

    # ------------------------------------------------------------------ #
    #  Terminal state checks                                               #
    # ------------------------------------------------------------------ #

    def is_terminal(self) -> bool:
        return self._winner is not None or self._is_draw

    @property
    def winner(self) -> Optional[int]:
        return self._winner

    @property
    def is_draw(self) -> bool:
        return self._is_draw

    def _check_win(self, row: int, col: int) -> bool:
        """
        Check whether the piece just placed at (row, col) creates a win.
        Tests all four directions from that cell.
        Faster than rescanning the full board each time.
        """
        player = self.grid[row][col]
        directions = [
            (0, 1),   # horizontal →
            (1, 0),   # vertical   ↓
            (1, 1),   # diagonal   ↘
            (1, -1),  # diagonal   ↙
        ]
        for dr, dc in directions:
            count = 1  # count the piece just placed
            # scan in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player:
                count += 1
                r += dr; c += dc
            # scan in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player:
                count += 1
                r -= dr; c -= dc
            if count >= 4:
                return True
        return False

    def winning_cells(self) -> List[Tuple[int, int]]:
        """
        Return the list of (row, col) cells that form the winning 4-in-a-row.
        Used by the visualizer to highlight the winning line.
        Returns [] if no winner yet.
        """
        if self._winner is None or self.last_move is None:
            return []
        row0, col0 = self.last_move
        # last_move was the winner's last piece — but after drop() current has
        # already been switched, so winner is the *previous* current.
        # We stored _winner before switching, so self.grid[row0][col0] == _winner.
        player = self._winner
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in directions:
            line = [(row0, col0)]
            for sign in (1, -1):
                r, c = row0 + sign*dr, col0 + sign*dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] == player:
                    line.append((r, c))
                    r += sign*dr; c += sign*dc
            if len(line) >= 4:
                return line
        return []

    # ------------------------------------------------------------------ #
    #  Heuristic evaluation (used by both MCTS rollout & Minimax)         #
    # ------------------------------------------------------------------ #

    def evaluate(self, player: int) -> float:
        """
        Static board evaluation for `player`.
        Returns a score in roughly [-1000, +1000].

        Strategy:
          1. Terminal states: ±1000
          2. Center column control: pieces in the center are more connected
          3. Count 'windows' of 4 cells and score them by composition
        """
        opponent = P2 if player == P1 else P1

        if self._winner == player:
            return 1000.0
        if self._winner == opponent:
            return -1000.0
        if self._is_draw:
            return 0.0

        score = 0.0

        # --- Center column preference ---
        # Center column (col=3) is the most valuable; score ownership there
        center_col = [int(self.grid[r][3]) for r in range(ROWS)]
        score += center_col.count(player)   * 6
        score -= center_col.count(opponent) * 6

        # Adjacent center columns (2 and 4)
        for col in [2, 4]:
            col_arr = [int(self.grid[r][col]) for r in range(ROWS)]
            score += col_arr.count(player)   * 3
            score -= col_arr.count(opponent) * 3

        # --- Window scoring ---
        # Slide a window of 4 over every row, column, and diagonal
        for window in self._all_windows():
            score += self._score_window(window, player)

        return score

    def _score_window(self, window: list, player: int) -> float:
        """Score a 4-cell window for `player`."""
        opponent = P2 if player == P1 else P1
        p_count  = window.count(player)
        o_count  = window.count(opponent)
        e_count  = window.count(EMPTY)

        if o_count > 0:
            return 0.0   # opponent blocks this window

        if p_count == 4: return 100.0
        if p_count == 3 and e_count == 1: return 10.0
        if p_count == 2 and e_count == 2: return 2.0
        return 0.0

    def _all_windows(self):
        """Generate all 4-cell windows on the board."""
        g = self.grid
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                yield [int(g[r][c+i]) for i in range(4)]
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                yield [int(g[r+i][c]) for i in range(4)]
        # Diagonal ↘
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                yield [int(g[r+i][c+i]) for i in range(4)]
        # Diagonal ↙
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                yield [int(g[r+i][c-i]) for i in range(4)]

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        symbols = {EMPTY: '.', P1: 'R', P2: 'Y'}
        lines = []
        for row in self.grid:
            lines.append(' '.join(symbols[c] for c in row))
        lines.append('-' * (COLS * 2 - 1))
        lines.append(' '.join(str(c) for c in range(COLS)))
        return '\n'.join(lines)


# ── Quick sanity check ─────────────────────────────────────────────────
if __name__ == '__main__':
    b = ConnectFourBoard()
    for col in [3, 3, 4, 4, 5, 5]:
        b.drop(col)
    print(b)
    print('Legal moves:', b.legal_moves())
    print('Terminal:', b.is_terminal())