"""
Goal-based agent for the Warehouse Navigation Problem (CS F407 - Agents Lab).

The agent keeps an explicit goal (reach G), an internal model of the
warehouse, and chooses actions by *planning* a collision-free path with
Breadth-First Search (BFS) before it moves.  It then executes the plan one
action at a time, checking after every step that the goal test still holds.

Why BFS?
    * Every move (Up/Down/Left/Right) costs exactly 1.
    * With uniform step costs, BFS expands states in order of depth, so the
      first time it reaches G the path is guaranteed to be a shortest one
      (BFS is complete and optimal for unit-cost problems).
    * The grid is small (21 x 7), so BFS's O(|S|) memory is not a concern.

Usage:
    python warehouse_agent.py                 # run on the lab map
    python warehouse_agent.py path/to/map.txt # run on any map file
"""

from __future__ import annotations

import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

Position = Tuple[int, int]  # (row, col)

WAREHOUSE_MAP = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# Action name -> (d_row, d_col)
ACTIONS: Dict[str, Position] = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class Warehouse:
    """The environment: a static, fully observable, deterministic grid."""

    def __init__(self, text: str):
        lines = [line.rstrip("\n") for line in text.strip("\n").splitlines()]
        if not lines:
            raise ValueError("Empty map")
        width = max(len(line) for line in lines)
        # Pad ragged rows with walls so every row has the same width.
        self.grid: List[List[str]] = [list(line.ljust(width, "#")) for line in lines]
        self.rows, self.cols = len(self.grid), width
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol: str) -> Position:
        found = [(r, c) for r in range(self.rows) for c in range(self.cols)
                 if self.grid[r][c] == symbol]
        if len(found) != 1:
            raise ValueError(f"Map must contain exactly one '{symbol}', found {len(found)}")
        return found[0]

    def is_free(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != "#"

    def result(self, pos: Position, action: str) -> Position:
        """Transition model: where an action leads (stays put if blocked)."""
        dr, dc = ACTIONS[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        return nxt if self.is_free(nxt) else pos

    def render(self, path: Optional[List[Position]] = None) -> str:
        grid = [row[:] for row in self.grid]
        for (r, c) in path or []:
            if grid[r][c] == ".":
                grid[r][c] = "*"
        return "\n".join("".join(row) for row in grid)


# --------------------------------------------------------------------------
# Goal-based agent
# --------------------------------------------------------------------------
class GoalBasedAgent:
    """
    Components (goal-based architecture from the lecture):
        state   - current position + the plan still to execute
        goal    - the target cell G
        model   - knowledge of how actions change the state (Warehouse.result)
        planner - BFS search that picks actions leading to the goal
    """

    def __init__(self, env: Warehouse):
        self.env = env
        self.position: Position = env.start
        self.goal: Position = env.goal
        self.plan: List[str] = []
        self.expanded = 0

    # ---- goal test -------------------------------------------------------
    def goal_reached(self) -> bool:
        return self.position == self.goal

    # ---- decision-making component --------------------------------------
    def search(self) -> Optional[List[str]]:
        """BFS over positions. Returns a list of actions, or None."""
        start = self.position
        frontier = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Optional[str]]] = {start: (None, None)}
        self.expanded = 0

        while frontier:
            current = frontier.popleft()
            self.expanded += 1
            if current == self.goal:
                return self._reconstruct(parent, current)
            for action in ACTIONS:
                nxt = self.env.result(current, action)
                if nxt != current and nxt not in parent:  # skip walls and visited
                    parent[nxt] = (current, action)
                    frontier.append(nxt)
        return None

    @staticmethod
    def _reconstruct(parent, node) -> List[str]:
        actions: List[str] = []
        while parent[node][0] is not None:
            prev, action = parent[node]
            actions.append(action)
            node = prev
        return actions[::-1]

    # ---- agent program ---------------------------------------------------
    def act(self) -> Optional[str]:
        """Return the next action towards the goal (plans lazily)."""
        if self.goal_reached():
            return None
        if not self.plan:
            plan = self.search()
            if plan is None:
                return None
            self.plan = plan
        return self.plan.pop(0)

    def run(self, max_steps: int = 10_000) -> Tuple[bool, List[Position], List[str]]:
        """Perceive-decide-act loop. Returns (success, visited cells, actions)."""
        trajectory = [self.position]
        actions: List[str] = []
        for _ in range(max_steps):
            if self.goal_reached():
                return True, trajectory, actions
            action = self.act()
            if action is None:
                return False, trajectory, actions
            new_pos = self.env.result(self.position, action)
            if new_pos == self.position:  # model mismatch -> replan
                self.plan = []
                continue
            self.position = new_pos
            trajectory.append(new_pos)
            actions.append(action)
        return self.goal_reached(), trajectory, actions


def solve(text: str) -> Tuple[bool, List[Position], List[str], int]:
    env = Warehouse(text)
    agent = GoalBasedAgent(env)
    ok, traj, acts = agent.run()
    return ok, traj, acts, agent.expanded


def main(argv: List[str]) -> int:
    text = open(argv[1]).read() if len(argv) > 1 else WAREHOUSE_MAP
    env = Warehouse(text)
    agent = GoalBasedAgent(env)
    ok, traj, acts = agent.run()

    print("Warehouse:")
    print(env.render())
    print(f"\nStart S = {env.start}, Goal G = {env.goal}  (row, col)")
    if not ok:
        print("\nNo collision-free path exists from S to G.")
        return 1
    print(f"\nPath found: {len(acts)} moves, {agent.expanded} states expanded by BFS")
    print("Actions :", " -> ".join(acts))
    print("Cells   :", " -> ".join(map(str, traj)))
    print("\nPath on map (* = route):")
    print(env.render(traj))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
