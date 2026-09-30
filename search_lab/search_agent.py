"""
Search agent for the warehouse robot (CS F407 - Search Lab).

Search problem P = (S, A, T, s0, G, c):
    S  : free grid cells (row, col)
    A  : {Up, Down, Left, Right}
    T  : T(s, a) = s + delta(a) if that cell is inside the grid and not '#'
    s0 : the cell marked 'S'
    G  : {the cell marked 'G'}
    c  : 1 per move

Implements A* (with a pluggable heuristic) and BFS, both reporting
    found?, path, path length (moves), states expanded.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Dict, List, Optional, Tuple

State = Tuple[int, int]  # (row, col)

ACTIONS: Dict[str, State] = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}

LAB_MAP = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""


# ---------------------------------------------------------------- problem
class GridProblem:
    def __init__(self, text: str):
        lines = text.strip("\n").splitlines()
        width = max(map(len, lines))
        self.grid = [line.ljust(width, "#") for line in lines]
        self.rows, self.cols = len(self.grid), width
        self.initial = self._locate("S")
        self.goal = self._locate("G")

    def _locate(self, ch: str) -> State:
        hits = [(r, c) for r, row in enumerate(self.grid) for c, x in enumerate(row) if x == ch]
        if len(hits) != 1:
            raise ValueError(f"map must contain exactly one '{ch}'")
        return hits[0]

    def is_goal(self, s: State) -> bool:          # goal test
        return s == self.goal

    def actions(self, s: State) -> List[str]:      # valid actions
        return [a for a in ACTIONS if self._free(self._move(s, a))]

    def result(self, s: State, a: str) -> State:   # transition T
        return self._move(s, a)

    def cost(self, s: State, a: str, s2: State) -> int:
        return 1

    @staticmethod
    def _move(s: State, a: str) -> State:
        dr, dc = ACTIONS[a]
        return (s[0] + dr, s[1] + dc)

    def _free(self, s: State) -> bool:
        r, c = s
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != "#"

    def render(self, path: Optional[List[State]] = None) -> str:
        g = [list(row) for row in self.grid]
        for r, c in path or []:
            if g[r][c] == ".":
                g[r][c] = "*"
        return "\n".join("".join(row) for row in g)


# ---------------------------------------------------------------- heuristics
Heuristic = Callable[[State, State], float]


def manhattan(s: State, g: State) -> float:
    return abs(s[0] - g[0]) + abs(s[1] - g[1])


def zero(s: State, g: State) -> float:
    return 0


def euclidean(s: State, g: State) -> float:
    return math.hypot(s[0] - g[0], s[1] - g[1])


def weighted(h: Heuristic, w: float) -> Heuristic:
    def hw(s: State, g: State) -> float:
        return w * h(s, g)
    hw.__name__ = f"{w:g}x{h.__name__}"
    return hw


HEURISTICS: Dict[str, Heuristic] = {
    "manhattan": manhattan,
    "zero": zero,
    "euclidean": euclidean,
    "2x_manhattan": weighted(manhattan, 2),
}


# ---------------------------------------------------------------- result
@dataclass
class SearchResult:
    algorithm: str
    found: bool
    path: List[State] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    expanded: int = 0

    @property
    def length(self) -> Optional[int]:
        return len(self.actions) if self.found else None


def _reconstruct(parent: Dict[State, Tuple[Optional[State], Optional[str]]], node: State):
    path, acts = [node], []
    while parent[node][0] is not None:
        node, a = parent[node]
        path.append(node)
        acts.append(a)
    return path[::-1], acts[::-1]


# ---------------------------------------------------------------- A*
def astar(problem: GridProblem, h: Heuristic = manhattan) -> SearchResult:
    start, goal = problem.initial, problem.goal
    tie = count()                                   # FIFO tie-breaking among equal f
    g_cost: Dict[State, float] = {start: 0}
    parent: Dict[State, Tuple[Optional[State], Optional[str]]] = {start: (None, None)}
    frontier = [(h(start, goal), next(tie), start)]  # priority queue ordered by f = g + h
    closed = set()                                  # visited / expanded states
    expanded = 0

    while frontier:
        f, _, s = heapq.heappop(frontier)           # select state with lowest f(n)
        if s in closed:                             # stale duplicate entry
            continue
        closed.add(s)
        expanded += 1
        if problem.is_goal(s):
            path, acts = _reconstruct(parent, s)
            return SearchResult(f"A*[{h.__name__}]", True, path, acts, expanded)
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            g2 = g_cost[s] + problem.cost(s, a, s2)  # g(n)
            if s2 not in closed and g2 < g_cost.get(s2, math.inf):
                g_cost[s2] = g2
                parent[s2] = (s, a)
                f2 = g2 + h(s2, goal)               # f(n) = g(n) + h(n)
                heapq.heappush(frontier, (f2, next(tie), s2))
    return SearchResult(f"A*[{h.__name__}]", False, expanded=expanded)


# ---------------------------------------------------------------- BFS
def bfs(problem: GridProblem) -> SearchResult:
    start = problem.initial
    parent: Dict[State, Tuple[Optional[State], Optional[str]]] = {start: (None, None)}
    frontier = deque([start])
    expanded = 0
    while frontier:
        s = frontier.popleft()
        expanded += 1
        if problem.is_goal(s):
            path, acts = _reconstruct(parent, s)
            return SearchResult("BFS", True, path, acts, expanded)
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            if s2 not in parent:
                parent[s2] = (s, a)
                frontier.append(s2)
    return SearchResult("BFS", False, expanded=expanded)


# ---------------------------------------------------------------- CLI
def report(problem: GridProblem, res: SearchResult) -> str:
    lines = [f"{res.algorithm}: found={res.found}  length={res.length}  expanded={res.expanded}"]
    if res.found:
        lines.append("path: " + " -> ".join(map(str, res.path)))
        lines.append(problem.render(res.path))
    else:
        lines.append("No path from S to G.")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    text = open(argv[1]).read() if len(argv) > 1 else LAB_MAP
    p = GridProblem(text)
    print(report(p, astar(p, manhattan)))
    print()
    print(report(p, bfs(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
