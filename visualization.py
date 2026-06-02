"""
visualization.py
════════════════
All rendering logic for Connect Four.
Completely decoupled from the agents and game engine.

Functions
---------
draw_board(board, title, ax)          — render a single board state
animate_game(history, interval)       — animate a full game move-by-move
draw_mcts_tree(root, max_depth)       — render the MCTS decision tree
draw_uct_analysis(root)               — UCT breakdown for root children
draw_comparison(results)              — bar/violin comparison charts
"""

import math
import random
import numpy as np
import matplotlib
from mcts_agent import MCTSAgent, MCTSNode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch
from collections import defaultdict

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from connect_four import ConnectFourBoard, ROWS, COLS, P1, P2, EMPTY


# ── Palette ───────────────────────────────────────────────────────────
BG      = '#0d0d14'
PAN     = '#13131e'
BOARD   = '#1a3a5c'       # deep blue — the classic C4 board color
EMPTY_C = '#0d1f33'       # empty cell hole
P1_C    = '#e63946'       # Red  player
P2_C    = '#f7c948'       # Yellow player
WIN_C   = '#ffffff'       # winning disc highlight
GOLD    = '#f7c948'
TEAL    = '#38d9a9'
RED_C   = '#e63946'
BLUE    = '#74b9ff'
WHITE   = '#e8e0c8'
GRAY    = '#6a6a8a'

PLAYER_COLORS = {P1: P1_C, P2: P2_C, EMPTY: EMPTY_C}
PLAYER_NAMES  = {P1: 'Red', P2: 'Yellow'}


# ════════════════════════════════════════════════════════════════════ #
#  1. Board Renderer                                                    #
# ════════════════════════════════════════════════════════════════════ #

def draw_board(board: ConnectFourBoard,
               title: str = '',
               ax=None,
               highlight_last: bool = True,
               show_col_labels: bool = True):
    """
    Draw a beautiful Connect Four board on `ax` (or create a new figure).

    Visual design
    -------------
    - Dark navy board background
    - Circular cell holes with colored discs
    - Last-placed piece has a bright white ring
    - Winning 4-in-a-row pulses with a white outline
    - Column numbers along the bottom
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor(BG)

    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.8, ROWS + 0.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Board background rectangle
    board_rect = plt.Rectangle((-0.55, -0.55), COLS + 0.1, ROWS + 0.1,
                                color=BOARD, zorder=0, linewidth=0,
                                edgecolor=None, alpha=0.95)
    ax.add_patch(board_rect)

    # Winning cells
    win_cells = set(board.winning_cells())

    # Draw cells
    for r in range(ROWS):
        for c in range(COLS):
            val     = board.grid[r][c]
            display_row = ROWS - 1 - r   # flip: row 0 (top) → displayed at top

            # Disc color
            color = PLAYER_COLORS[val]

            # Cell shadow (makes board feel 3D)
            shadow = Circle((c, display_row), 0.43,
                            color='#000000', alpha=0.3, zorder=1)
            ax.add_patch(shadow)

            # Main disc
            disc = Circle((c, display_row), 0.40,
                          color=color, zorder=2)
            ax.add_patch(disc)

            # Inner shine (empty cells look like holes, filled ones like discs)
            if val == EMPTY:
                shine = Circle((c - 0.08, display_row + 0.08), 0.10,
                               color='#1a4a7a', alpha=0.5, zorder=3)
            else:
                shine = Circle((c - 0.10, display_row + 0.10), 0.10,
                               color='white', alpha=0.25, zorder=3)
            ax.add_patch(shine)

            # Highlight last move
            if highlight_last and board.last_move == (r, c) and val != EMPTY:
                ring = Circle((c, display_row), 0.42,
                              fill=False, edgecolor='white',
                              linewidth=2.5, zorder=4)
                ax.add_patch(ring)

            # Highlight winning cells
            if (r, c) in win_cells:
                win_ring = Circle((c, display_row), 0.44,
                                  fill=False, edgecolor=WIN_C,
                                  linewidth=3.5, zorder=5, alpha=0.95)
                ax.add_patch(win_ring)

    # Column numbers
    if show_col_labels:
        for c in range(COLS):
            ax.text(c, -0.62, str(c), ha='center', va='center',
                    fontsize=10, color=GRAY, fontfamily='monospace')

    # Title
    if title:
        ax.set_title(title, color=GOLD, fontsize=12,
                     fontfamily='monospace', pad=8)

    # Status text
    if board.is_terminal():
        if board.winner:
            status = f"🏆  {PLAYER_NAMES[board.winner]} wins!"
            color  = PLAYER_COLORS[board.winner]
        else:
            status = "Draw!"
            color  = WHITE
        ax.text(COLS / 2 - 0.5, ROWS + 0.05, status,
                ha='center', va='bottom', fontsize=13,
                color=color, fontweight='bold', fontfamily='monospace')
    else:
        turn_color = PLAYER_COLORS[board.current]
        ax.text(COLS / 2 - 0.5, ROWS + 0.05,
                f"▶  {PLAYER_NAMES[board.current]}'s turn  (move {board.move_count + 1})",
                ha='center', va='bottom', fontsize=10,
                color=turn_color, fontfamily='monospace')

    if standalone:
        plt.tight_layout()
        plt.show()


# ════════════════════════════════════════════════════════════════════ #
#  2. Game Animator                                                     #
# ════════════════════════════════════════════════════════════════════ #

def animate_game(history: list, interval: int = 700,
                 title: str = 'Connect Four', figsize=(8, 7)):
    """
    Animate a game from a list of ConnectFourBoard states.

    Parameters
    ----------
    history  : list of ConnectFourBoard — one state per move
    interval : ms between frames
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)

    def update(frame):
        ax.clear()
        board = history[frame]
        move_label = f"Move {frame}" if frame > 0 else "Start"
        draw_board(board, title=f'{title}  |  {move_label}', ax=ax)

    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=interval, repeat=False)
    plt.tight_layout()
    return anim


# ════════════════════════════════════════════════════════════════════ #
#  3. MCTS Tree Visualizer                                              #
# ════════════════════════════════════════════════════════════════════ #

def draw_mcts_tree(root, max_depth: int = 3,
                   title: str = 'MCTS Decision Tree',
                   figsize=(14, 7)):
    """
    Visualize the MCTS search tree after a decision.

    Visual encoding
    ---------------
    - Node SIZE   ∝ visit count N (more visited = larger circle)
    - Node COLOR  = Q/N win rate  (gold=high, teal=mid, red=low)
    - GOLD edges  = best path (most-visited chain from root)
    - Labels show: column choice, visit count, win rate
    """
    if not HAS_NX:
        print("networkx not installed — pip install networkx")
        return

    # Build NetworkX graph
    G = nx.DiGraph()

    def add_node(node, depth=0):
        if depth > max_depth:
            return
        G.add_node(node.id,
                   N     = node.N,
                   Q     = node.value,
                   depth = node.depth,
                   move  = node.move)
        for child in node.children:
            if child.depth <= max_depth:
                G.add_edge(node.id, child.id)
                add_node(child, depth + 1)

    add_node(root)

    if len(G.nodes) == 0:
        return

    # Layout: hierarchical by depth
    pos = _hierarchy_pos(G, root.id)

    # Node styling
    node_list = list(G.nodes())
    max_N     = max(G.nodes[n]['N'] for n in node_list) or 1
    sizes     = [150 + 1200 * (G.nodes[n]['N'] / max_N) for n in node_list]

    vals   = [G.nodes[n]['Q'] for n in node_list]
    max_Q  = max(vals) if vals else 1
    colors = []
    for v in vals:
        t = v / max(max_Q, 0.001)
        if t > 0.65:   colors.append(GOLD)
        elif t > 0.35: colors.append(TEAL)
        else:          colors.append(RED_C)

    # Best path (most-visited chain)
    best_edges = set()
    cur = root
    while cur.children:
        mv = cur.most_visited()
        best_edges.add((cur.id, mv.id))
        cur = mv

    edge_colors = [GOLD if (u, v) in best_edges else '#2a2a50'
                   for u, v in G.edges()]
    edge_widths = [2.8  if (u, v) in best_edges else 0.7
                   for u, v in G.edges()]

    # Draw
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PAN)
    ax.axis('off')

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           width=edge_widths,
                           arrows=True, arrowstyle='->',
                           arrowsize=12, alpha=0.9)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors,
                           node_size=sizes,
                           alpha=0.92)

    # Labels
    for n in node_list:
        x, y  = pos[n]
        nd    = G.nodes[n]
        label = f"col {nd['move']}" if nd['move'] is not None else 'ROOT'
        ax.text(x, y + 0.015, label,
                ha='center', va='center',
                fontsize=7, color='#0d0d14',
                fontweight='bold', fontfamily='monospace')
        ax.text(x, y - 0.05, f"N={nd['N']}  Q={nd['Q']:.2f}",
                ha='center', va='center',
                fontsize=5.5, color='#0d0d14',
                fontfamily='monospace')

    # Legend
    handles = [
        mpatches.Patch(color=GOLD,  label='Win rate > 65%'),
        mpatches.Patch(color=TEAL,  label='Win rate 35–65%'),
        mpatches.Patch(color=RED_C, label='Win rate < 35%'),
        plt.Line2D([0], [0], color=GOLD, lw=2.5, label='Best path'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              facecolor=PAN, edgecolor=GOLD, labelcolor=WHITE, framealpha=0.9)

    total = sum(1 for _ in _all_nodes(root))
    ax.set_title(f'{title}  |  {total} total nodes  |  depth shown: {max_depth}',
                 color=GOLD, fontsize=12, fontfamily='monospace', pad=10)
    plt.tight_layout()
    plt.show()


def _all_nodes(root):
    yield root
    for c in root.children:
        yield from _all_nodes(c)


def _hierarchy_pos(G, root_id, vert_gap=0.28, vert_loc=0.9):
    pos = {}
    by_depth = defaultdict(list)

    def collect(nid, d):
        by_depth[d].append(nid)
        for s in G.successors(nid):
            collect(s, d + 1)

    collect(root_id, 0)

    for d, nodes in by_depth.items():
        y = vert_loc - d * vert_gap
        for i, n in enumerate(nodes):
            pos[n] = ((i + 1) / (len(nodes) + 1), y)
    return pos


# ════════════════════════════════════════════════════════════════════ #
#  4. UCT Child Analysis                                               #
# ════════════════════════════════════════════════════════════════════ #

def draw_uct_analysis(root, figsize=(10, 5)):
    """
    Bar chart breakdown of root children: visits, win rate, UCT score.
    """
    if not root.children:
        return

    children = sorted(root.children, key=lambda c: c.N, reverse=True)
    cols     = [f"col {c.move}" for c in children]
    visits   = [c.N     for c in children]
    winrates = [c.value for c in children]
    ucts     = [c.uct() if c.N > 0 else 0 for c in children]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor(BG)
    fig.suptitle('MCTS Root Children — Decision Breakdown',
                 color=GOLD, fontsize=13, fontfamily='monospace')

    bar_color = [GOLD if i == 0 else TEAL for i in range(len(cols))]

    # Visits
    ax = axes[0]; ax.set_facecolor(PAN)
    bars = ax.bar(cols, visits, color=bar_color, edgecolor='white',
                  linewidth=0.5, alpha=0.88)
    for b, v in zip(bars, visits):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                str(v), ha='center', fontsize=9, color=WHITE,
                fontfamily='monospace')
    ax.set_title('Visit Count (N)', color=GOLD, fontsize=10)
    ax.set_ylabel('N', color=WHITE); ax.tick_params(colors=WHITE)
    ax.set_facecolor(PAN); ax.grid(axis='y', alpha=0.2)

    # Win rate
    ax2 = axes[1]; ax2.set_facecolor(PAN)
    bars2 = ax2.bar(cols, [w * 100 for w in winrates],
                    color=bar_color, edgecolor='white',
                    linewidth=0.5, alpha=0.88)
    for b, v in zip(bars2, winrates):
        ax2.text(b.get_x() + b.get_width()/2,
                 b.get_height() + 0.5,
                 f'{v*100:.1f}%', ha='center', fontsize=9,
                 color=WHITE, fontfamily='monospace')
    ax2.set_title('Win Rate (Q/N)', color=GOLD, fontsize=10)
    ax2.set_ylabel('Win %', color=WHITE); ax2.tick_params(colors=WHITE)
    ax2.grid(axis='y', alpha=0.2)

    # UCT
    ax3 = axes[2]; ax3.set_facecolor(PAN)
    bars3 = ax3.bar(cols, ucts, color=bar_color, edgecolor='white',
                    linewidth=0.5, alpha=0.88)
    for b, v in zip(bars3, ucts):
        ax3.text(b.get_x() + b.get_width()/2,
                 b.get_height() + 0.002,
                 f'{v:.3f}', ha='center', fontsize=8,
                 color=WHITE, fontfamily='monospace')
    ax3.set_title('UCT Score', color=GOLD, fontsize=10)
    ax3.set_ylabel('UCT', color=WHITE); ax3.tick_params(colors=WHITE)
    ax3.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════ #
#  5. Tree Growth (iterations comparison)                              #
# ════════════════════════════════════════════════════════════════════ #

def draw_tree_growth(board, iter_counts=(5, 20, 100, 500), figsize=(18, 5)):
    """
    Show how the MCTS tree evolves with more iterations.
    """
    
    fig, axes = plt.subplots(1, len(iter_counts), figsize=figsize)
    fig.patch.set_facecolor(BG)
    fig.suptitle('MCTS Tree Growth — More Iterations = Better Decisions',
                 color=GOLD, fontsize=13, fontfamily='monospace')

    for ax, n_it in zip(axes, iter_counts):
        MCTSNode._id_counter = 0
        _, root_tmp = MCTSAgent(n_iterations=n_it).search(board)

        if not HAS_NX:
            ax.text(0.5, 0.5, 'networkx missing', ha='center',
                    color=WHITE, transform=ax.transAxes)
            continue

        G   = _build_nx(root_tmp, max_depth=3)
        pos = _hierarchy_pos(G, 0)
        nls = list(G.nodes())
        if not nls:
            ax.axis('off'); continue

        mN  = max(G.nodes[n]['N'] for n in nls) or 1
        sz  = [120 + 800 * (G.nodes[n]['N'] / mN) for n in nls]
        vl  = [G.nodes[n]['Q'] for n in nls]
        mQ  = max(vl) if vl else 1
        nc  = [GOLD if v/max(mQ,0.001) > 0.65 else
               TEAL if v/max(mQ,0.001) > 0.35 else
               RED_C for v in vl]

        best_e = set()
        cur = root_tmp
        while cur.children:
            mv = cur.most_visited()
            best_e.add((cur.id, mv.id))
            cur = mv

        ec = [GOLD if (u,v) in best_e else '#2a2a50' for u,v in G.edges()]
        ew = [2.5  if (u,v) in best_e else 0.6       for u,v in G.edges()]

        ax.set_facecolor(PAN); ax.axis('off')
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ec, width=ew,
                               arrows=True, arrowstyle='->', arrowsize=8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc,
                               node_size=sz, alpha=0.9)
        lbs = {n: (f"col {G.nodes[n]['move']}"
                   if G.nodes[n]['move'] is not None else 'ROOT')
               for n in nls}
        nx.draw_networkx_labels(G, pos, lbs, ax=ax, font_size=5.5,
                                font_color='#0d0d14', font_weight='bold')
        total = sum(1 for _ in _all_nodes(root_tmp))
        ax.set_title(f'{n_it} iterations  |  {total} nodes',
                     fontsize=9, color=WHITE, fontfamily='monospace')

    plt.tight_layout()
    plt.show()


def _build_nx(root, max_depth=3):
    G = nx.DiGraph()

    def add(node):
        if node.depth > max_depth:
            return
        G.add_node(node.id, N=node.N, Q=node.value,
                   depth=node.depth, move=node.move)
        for c in node.children:
            if c.depth <= max_depth:
                G.add_edge(node.id, c.id)
                add(c)

    add(root)
    return G


# ════════════════════════════════════════════════════════════════════ #
#  6. Match Comparison Chart                                           #
# ════════════════════════════════════════════════════════════════════ #

def draw_comparison(results: dict, figsize=(14, 5)):
    """
    Draw win/loss/draw comparison from a tournament results dict.

    results format:
    {
      'MCTS':    {'wins': 12, 'losses': 5, 'draws': 3},
      'Minimax': {'wins':  5, 'losses': 12, 'draws': 3},
      ...
    }
    """
    agents   = list(results.keys())
    wins_pct = [results[a]['wins']   / sum(results[a].values()) * 100 for a in agents]
    draw_pct = [results[a]['draws']  / sum(results[a].values()) * 100 for a in agents]
    loss_pct = [results[a]['losses'] / sum(results[a].values()) * 100 for a in agents]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(BG)
    fig.suptitle('Tournament Results', color=GOLD,
                 fontsize=13, fontfamily='monospace')

    # Stacked bar
    ax = axes[0]; ax.set_facecolor(PAN)
    x = range(len(agents))
    b1 = ax.bar(x, wins_pct, color=TEAL,  label='Win',  alpha=0.88)
    b2 = ax.bar(x, draw_pct, bottom=wins_pct, color=GRAY, label='Draw', alpha=0.88)
    b3 = ax.bar(x, loss_pct, bottom=[w+d for w,d in zip(wins_pct, draw_pct)],
                color=RED_C, label='Loss', alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(agents, color=WHITE, fontsize=10)
    ax.set_ylabel('Percentage (%)', color=WHITE)
    ax.set_title('Win / Draw / Loss Rate', color=GOLD, fontsize=10)
    ax.legend(facecolor=PAN, labelcolor=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE); ax.grid(axis='y', alpha=0.2)
    for b, v in zip(b1, wins_pct):
        ax.text(b.get_x() + b.get_width()/2, v/2,
                f'{v:.0f}%', ha='center', color='#0d0d14',
                fontweight='bold', fontsize=10)

    # Win count bars
    ax2 = axes[1]; ax2.set_facecolor(PAN)
    win_counts = [results[a]['wins'] for a in agents]
    cols_bar   = [GOLD if w == max(win_counts) else BLUE for w in win_counts]
    bars = ax2.bar(agents, win_counts, color=cols_bar,
                   edgecolor='white', linewidth=0.5, alpha=0.88)
    for b, v in zip(bars, win_counts):
        ax2.text(b.get_x() + b.get_width()/2,
                 b.get_height() + 0.1,
                 str(v), ha='center', fontsize=12,
                 color=WHITE, fontweight='bold', fontfamily='monospace')
    ax2.set_title('Total Wins', color=GOLD, fontsize=10)
    ax2.set_ylabel('Wins', color=WHITE)
    ax2.tick_params(colors=WHITE); ax2.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.show()