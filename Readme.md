# Connect Four — MCTS vs Minimax
**Foundations of Adversarial Search — From Minimax to Monte Carlo Tree Search**

A from-scratch implementation of two adversarial AI agents — Monte Carlo Tree Search (MCTS) 
and Minimax with Alpha-Beta pruning — competing against each other on a fully custom Connect 
Four engine. The project explores how statistical sampling (MCTS) and exhaustive game tree 
search (Minimax) approach the same problem from fundamentally different directions.

---

## Repository Structure

```
MCTS/
├── connect_four.py          # Game engine — rules, gravity, win detection
├── mcts_agent.py            # MCTS agent — 4-phase loop, UCT selection
├── minimax_agent.py         # Minimax agent — Alpha-Beta pruning
├── visualization.py         # Rendering — board, tree, analysis charts
├── connect_four_mcts.ipynb  # Main notebook — runs all experiments
├── outputs/                 # Generated figures
├── requirements.txt
└── README.md
```

Each module is fully decoupled. The game engine has no knowledge of the AI agents. The agents have no knowledge of rendering.

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook connect_four_mcts.ipynb
```

---

## 1. Game Engine

6×7 board with gravity physics, win detection in all 4 directions from the last placed piece, and full clone support for tree search without mutation side effects.

![Engine Check](outputs/engine_check.png)

Win detection verified in all 4 directions:

![Win Detection](outputs/win_detection.png)

---

## 2. MCTS Algorithm

Implements the standard 4-phase loop:

| Phase | Implementation |
|---|---|
| **Selection** | Traverse tree via UCT until an expandable or terminal node |
| **Expansion** | Add one child node for a randomly chosen untried move |
| **Simulation** | Center-weighted random rollout to terminal state |
| **Backpropagation** | Propagate reward up to root — `N += 1`, `Q += R` |

UCT formula (C = √2):
```
UCT(v) = Q(v)/N(v)  +  √2 · sqrt( ln(N(parent)) / N(v) )
```

Final move selection uses **most visits (N)**, not highest UCT — statistically more robust after N simulations.

### MCTS Decision Breakdown

Visit count, win rate, and UCT score for each column at the root:

![UCT Analysis](outputs/uct_analysis.png)

### MCTS Search Tree (600 iterations)

Node size ∝ visit count. Color = win rate. Gold path = best move chain.

![MCTS Tree](outputs/mcts_tree.png)

### Tree Growth — More Iterations = Better Decisions

As N → ∞, MCTS converges to the same solution as perfect Minimax:

![Tree Growth](outputs/tree_growth.png)

---

## 3. Exploration vs Exploitation

The UCT constant C controls how the agent balances known good moves against unexplored ones. C = √2 is theoretically optimal. UCT naturally allocates more visits to stronger columns without being told which ones they are.

![Exploration vs Exploitation](outputs/exploration.png)

---

## 4. MCTS vs Baselines

30 games each against a random opponent:

![Comparison](outputs/comparison.png)

MCTS achieves 100% win rate vs random. Greedy (center-first) achieves 90%. Random baseline 50%.

---

## 5. Bonus — Minimax + Alpha-Beta vs MCTS Live Match

Minimax assumes both players play perfectly:
- MAX picks the highest scoring move
- MIN picks the lowest scoring move for the opponent
- Alpha-Beta prunes branches that cannot affect the result
- Reduces complexity from O(b^d) to O(b^(d/2))

### Match Board — MCTS (Red) vs Minimax (Yellow)

![Match Board](outputs/match_board.png)

### MCTS Decision Tree at Move 1

![Match Tree](outputs/match_tree.png)

### Tournament Results — 10 Games

![Tournament](outputs/tournament.png)

Minimax (depth 5) wins all 10 games against MCTS (600 iterations). This is expected — Minimax at depth 5 has deterministic perfect-play guarantees that 600-iteration MCTS cannot yet match. Increasing MCTS iterations closes this gap as N → ∞.

---

## Key Takeaways

| | Minimax + α-β | MCTS |
|---|---|---|
| **Requires heuristic** | Yes | No |
| **Complexity** | O(b^d) → O(b^(d/2)) | O(N) any budget |
| **Optimal** | Yes, at sufficient depth | Converges as N → ∞ |
| **Anytime** | No | Yes |
| **Works for Go** | No | Yes (AlphaGo) |

---

## Author

Yonatan Azmir — May 2026