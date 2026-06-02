"""
minimax_agent.py
════════════════
Minimax agent with Alpha-Beta pruning for Connect Four.

Theory
------
Minimax assumes both players play perfectly:
  - MAX player (us)      → chooses the move with the HIGHEST score
  - MIN player (opponent)→ chooses the move with the LOWEST  score
  V(s) = max( min( V(s') ) )   — recurse from terminal leaves to root

Alpha-Beta pruning skips branches that cannot affect the final result:
  α = best score MAX is already guaranteed  (starts at -∞)
  β = best score MIN is already guaranteed  (starts at +∞)
  Prune when α ≥ β — this branch is irrelevant.

Complexity
----------
  Minimax:            O(b^d)      b=7, d=depth
  Alpha-Beta (best):  O(b^(d/2))  effectively doubles searchable depth

Move ordering: check center columns first — they are statistically
stronger and allow Alpha-Beta to prune more aggressively.
"""

from connect_four import ConnectFourBoard, P1, P2, COLS

# Preferred move order: center-out (maximizes pruning)
MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]


class MinimaxAgent:
    """
    Minimax agent with Alpha-Beta pruning.

    Parameters
    ----------
    depth   : int  — how many plies (half-moves) to search
    player  : int  — P1 or P2 (the agent plays as this player)
    """

    def __init__(self, depth: int = 5, player: int = P2):
        self.depth  = depth
        self.player = player
        self.nodes_searched = 0   # diagnostic counter

    def search(self, board: ConnectFourBoard) -> int:
        """
        Entry point. Returns the best column to play.
        """
        self.nodes_searched = 0
        _, move = self._minimax(board, self.depth,
                                alpha=float('-inf'),
                                beta=float('+inf'),
                                maximizing=True)
        return move

    def _minimax(self, board: ConnectFourBoard, depth: int,
                 alpha: float, beta: float, maximizing: bool):
        """
        Recursive minimax with Alpha-Beta pruning.

        Returns (score, best_move).
        """
        self.nodes_searched += 1

        # Terminal conditions
        if board.is_terminal():
            if board.winner == self.player:
                return (1_000_000 + depth, None)   # win sooner = better
            elif board.winner is not None:
                return (-1_000_000 - depth, None)  # lose later = better
            else:
                return (0, None)                    # draw

        if depth == 0:
            return (board.evaluate(self.player), None)

        legal = [m for m in MOVE_ORDER if m in board.legal_moves()]

        if maximizing:
            best_score = float('-inf')
            best_move  = legal[0]
            for move in legal:
                child = board.clone().drop(move)
                score, _ = self._minimax(child, depth - 1, alpha, beta,
                                         maximizing=False)
                if score > best_score:
                    best_score = score
                    best_move  = move
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break            # ✂ Beta cut-off — prune remaining
            return best_score, best_move

        else:  # minimizing
            best_score = float('+inf')
            best_move  = legal[0]
            for move in legal:
                child = board.clone().drop(move)
                score, _ = self._minimax(child, depth - 1, alpha, beta,
                                         maximizing=True)
                if score < best_score:
                    best_score = score
                    best_move  = move
                beta = min(beta, best_score)
                if alpha >= beta:
                    break            # ✂ Alpha cut-off — prune remaining
            return best_score, best_move


# ── Quick sanity check ─────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    board = ConnectFourBoard()
    agent = MinimaxAgent(depth=5, player=P2)
    t0 = time.time()
    move = agent.search(board)
    elapsed = time.time() - t0
    print(f"Minimax best move: col {move}  |  {elapsed*1000:.0f}ms  "
          f"|  nodes searched: {agent.nodes_searched:,}")