# Agents Lab: Problem Analysis and Agent Design

## Task 1: Understanding the Problem

| Question | Answer |
|---|---|
| **Environment** | A 7 × 21 warehouse grid. Cells are free (`.`), obstacles (`#`), start (`S`) or goal (`G`). It is fully observable, deterministic, static, discrete and single-agent. |
| **Goal** | Move the vehicle from `S` (row 1, col 1) to `G` (row 1, col 19) without entering an obstacle cell. |
| **Actions** | `Up`, `Down`, `Left`, `Right`. Each moves the vehicle one cell, and every move costs 1. |
| **Information to maintain** | The current position, the goal position, the map (transition model), the set of visited cells, and the plan (the sequence of actions still to execute). |
| **Why goal-based, not reflex** | A reflex agent maps the current percept to an action with fixed rules. At `S`, rules like "move right" lead into the wall at column 6, and nothing in the local percept says whether to go around above or below. A goal-based agent reasons about future states: it asks "which sequence of actions reaches G?" and searches before acting. |

### Think About It: a warehouse twice as large

BFS would still be correct and optimal, because every step still costs 1. Its time and memory grow
with the number of reachable cells, O(rows × cols). Doubling each dimension roughly quadruples the
work. For large maps, an informed search such as A* with the Manhattan heuristic expands far fewer
cells. Other difficulties appear in larger settings:

- moving obstacles such as other robots or people, which require replanning;
- several vehicles competing for the same aisles;
- partial observability, where the sensors do not show the whole map;
- non-uniform costs such as turns, congestion or battery use, which call for uniform-cost search or A*.

## Task 2: Agent Design

| Component | Realisation in `warehouse_agent.py` |
|---|---|
| Environment | `Warehouse`: the grid, `is_free()` and `result(pos, action)` (the transition model) |
| Current state | `GoalBasedAgent.position`, plus the remaining `plan` |
| Goal | `GoalBasedAgent.goal`, checked by `goal_reached()` |
| Actions | `ACTIONS = {Up, Down, Left, Right}` as (Δrow, Δcol) offsets |
| Decision-making | `search()`: breadth-first search over positions, which returns an action sequence |

### Block diagram

```
            +------------------------------------------------------+
            |                  GOAL-BASED AGENT                    |
 percept    |  +-------------+      +--------------------------+   |
 (position) |  |  State      |      |  Model of the world      |   |
 ---------->|  |  (row, col) |----->|  "what happens if I do   |   |
            |  +-------------+      |   action A in state s?"  |   |
            |         |             +------------+-------------+   |
            |         v                          v                 |
            |  +-------------+      +--------------------------+   |
            |  |  Goal       |----->|  Planner (BFS search)     |   |
            |  |  reach G    |      |  -> sequence of actions  |   |
            |  +-------------+      +------------+-------------+   |
            |                                    | next action     |
            +------------------------------------+-----------------+
                                                 v
                                   +---------------------------+
                                   |  ENVIRONMENT (warehouse)  |
                                   |  vehicle moves one cell   |
                                   +---------------------------+
```

### Algorithm choice

I chose **breadth-first search** before generating any code, for three reasons:

- Every step costs 1, so BFS returns a **shortest** path.
- BFS is **complete**: if a path exists it finds one, and if none exists it terminates and says so.
- The state space is tiny (under 100 free cells), so BFS's memory use does not matter.

### Validation criteria

A run counts as correct only if all of the following hold:

1. The path starts at `S`, ends at `G`, and every step moves exactly one cell into a free cell.
2. The path length equals the true shortest distance, checked against an independent BFS.
3. A map with no path makes the agent report failure instead of looping.
4. Edge cases work: the goal next to the start, several alternative routes, and malformed maps.
