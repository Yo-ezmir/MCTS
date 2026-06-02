"""
mcts_agent.py
═════════════
Monte Carlo Tree Search agent for Connect Four.

Core loop (repeated N times per decision):
  ① Selection   — walk tree via UCT until an expandable node
  ② Expansion   — add one new child for an untried move
  ③ Simulation  — random rollout to terminal state → reward
  ④ Backprop    — update N and Q for every node on the path

Final decision: pick the child with the MOST VISITS (not highest UCT).
Most-visits is more statistically robust than highest average for the
final choice — UCT is only for navigation during search.

UCT formula
-----------
  UCT(v) = Q(v)/N(v)  +  C · √( ln N(parent) / N(v) )
              ↑ exploit            ↑ explore
  C ≈ √2 is the theoretical optimum for rewards normalised to [0,1].
  We use C = 1.41 and scale rewards so they stay in a bounded range.
"""

import math
import random
from typing import Optional, List

from connect_four import ConnectFourBoard, P1, P2, COLS


# ── UCT exploration constant ───────────────────────────────────────────
C_UCT = math.sqrt(2)

# ── Rollout bias: weight toward center columns ─────────────────────────
# In Connect Four, center columns lead to more connections.
# We bias the random rollout toward them for stronger play.
CENTER_WEIGHTS = [1, 2, 3, 5, 3, 2, 1]   # col 0 → 6


class MCTSNode:
    """
    One node = one reachable board state in the MCTS tree.

    Attributes
    ----------
    N   : int   — visit count (how many simulations passed through here)
    Q   : float — total accumulated reward across all simulations
    """
    _id_counter = 0

    def __init__(self,
                 board:     ConnectFourBoard,
                 parent:    Optional["MCTSNode"] = None,
                 move:      Optional[int]        = None):
        self.id       = MCTSNode._id_counter
        MCTSNode._id_counter += 1

        self.board    = board          # game state at this node
        self.parent   = parent         # None for root
        self.move     = move           # column that produced this node
        self.children: List["MCTSNode"] = []

        self.N = 0                     # visit count
        self.Q = 0.0                   # total reward

        self.depth = parent.depth + 1 if parent else 0

        # Untried legal moves — shuffled to avoid ordering bias
        self.untried = board.legal_moves()
        random.shuffle(self.untried)

    # ── Value & UCT ───────────────────────────────────────────────────

    @property
    def value(self) -> float:
        """Average reward (Q/N). 0 if never visited."""
        return self.Q / self.N if self.N > 0 else 0.0

    def uct(self, C: float = C_UCT) -> float:
        """
        UCT score used during Selection.
        Unvisited nodes return +∞ so they are always tried first.
        """
        if self.N == 0:
            return float('inf')
        exploit = self.Q / self.N
        explore = C * math.sqrt(math.log(self.parent.N) / self.N)
        return exploit + explore

    # ── Tree navigation helpers ────────────────────────────────────────

    def is_fully_expanded(self) -> bool:
        """True when every legal move has a child node."""
        return len(self.untried) == 0

    def best_child_uct(self, C: float = C_UCT) -> "MCTSNode":
        """Return child with highest UCT — used during Selection."""
        return max(self.children, key=lambda c: c.uct(C))

    def most_visited(self) -> "MCTSNode":
        """Return child with most visits — used for the FINAL decision."""
        return max(self.children, key=lambda c: c.N)

    def __repr__(self) -> str:
        move_str = f"col={self.move}" if self.move is not None else "ROOT"
        return f"MCTSNode({move_str}  N={self.N}  Q/N={self.value:.2f})"


# ══════════════════════════════════════════════════════════════════════ #
#  MCTSAgent                                                             #
# ══════════════════════════════════════════════════════════════════════ #

class MCTSAgent:
    """
    MCTS agent. The tree is rebuilt from scratch at each decision step.

    Parameters
    ----------
    n_iterations : int   — simulations per move (more = stronger, slower)
    C            : float — UCT exploration constant
    """

    def __init__(self, n_iterations: int = 600, C: float = C_UCT):
        self.n_iter = n_iterations
        self.C      = C

    # ── Phase 1: Selection ─────────────────────────────────────────────
    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Walk down the tree following UCT until we reach:
          - a node that still has untried moves (expandable), OR
          - a terminal node.
        """
        node = root
        while not node.board.is_terminal():
            if not node.is_fully_expanded():
                return node          # still has untried children
            node = node.best_child_uct(self.C)
        return node                  # terminal leaf

    # ── Phase 2: Expansion ────────────────────────────────────────────
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Pick one untried move, apply it to a cloned board,
        and attach the resulting node as a new child.
        """
        if node.board.is_terminal() or not node.untried:
            return node
        move       = node.untried.pop()
        new_board  = node.board.clone().drop(move)
        child      = MCTSNode(new_board, parent=node, move=move)
        node.children.append(child)
        return child

    # ── Phase 3: Simulation (Rollout) ─────────────────────────────────
    def _simulate(self, node: MCTSNode, perspective: int) -> float:
        """
        From the node's board, play randomly to a terminal state.
        Returns a reward in {-1, 0, +1} from `perspective`'s viewpoint.

        Rollout policy: center-weighted random (not pure random).
        Center columns create more threats → biasing toward them
        makes rollouts more informative.
        """
        board = node.board.clone()
        while not board.is_terminal():
            moves   = board.legal_moves()
            weights = [CENTER_WEIGHTS[m] for m in moves]
            move    = random.choices(moves, weights=weights, k=1)[0]
            board.drop(move)

        if board.winner == perspective:
            return 1.0
        elif board.is_draw:
            return 0.5
        else:
            return 0.0

    # ── Phase 4: Backpropagation ───────────────────────────────────────
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Walk from the expanded node back to the root.
        At each node: N += 1,  Q += reward.
        Every ancestor gets credit for leading to this simulation.
        """
        cur = node
        while cur is not None:
            cur.N += 1
            cur.Q += reward
            cur    = cur.parent

    # ── Main search loop ───────────────────────────────────────────────
    def search(self, board: ConnectFourBoard) -> tuple:
        """
        Run `n_iterations` of the 4-phase MCTS loop from `board`.
        Returns (best_move, root_node).

        best_move is the column with the most visits — statistically
        the most reliable choice after N simulations.
        """
        MCTSNode._id_counter = 0
        perspective = board.current          # the player making the decision
        root        = MCTSNode(board.clone())

        for _ in range(self.n_iter):
            # ① Selection
            leaf   = self._select(root)
            # ② Expansion
            child  = self._expand(leaf)
            # ③ Simulation
            reward = self._simulate(child, perspective)
            # ④ Backpropagation
            self._backpropagate(child, reward)

        if not root.children:
            # Fallback: no children means terminal at root — pick any legal move
            return board.legal_moves()[0], root

        best = root.most_visited()
        return best.move, root


# ── Quick sanity check ─────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    board = ConnectFourBoard()
    agent = MCTSAgent(n_iterations=500)
    t0 = time.time()
    move, root = agent.search(board)
    elapsed = time.time() - t0
    print(f"Best move: col {move}  |  {elapsed*1000:.0f}ms  |  {root.N} root visits")
    print("Root children:")
    for c in sorted(root.children, key=lambda x: x.N, reverse=True):
        print(f"  col={c.move}  N={c.N}  Q/N={c.value:.3f}  UCT={c.uct():.3f}")