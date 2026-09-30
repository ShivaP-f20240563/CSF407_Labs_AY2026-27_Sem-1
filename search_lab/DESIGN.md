# Search Lab: Problem Formulation and Agent Design

## Task 0: The Warehouse as a Search Problem

| Component | Specification |
|---|---|
| **States S** | Every free cell `(row, col)` of the 9 × 17 grid, i.e. cells not marked `#`. The lab map has 64 of them. |
| **Actions A** | `{Up, Down, Left, Right}` |
| **Transition T** | `T((r,c), a) = (r+Δr, c+Δc)`, where Up = (−1,0), Down = (+1,0), Left = (0,−1), Right = (0,+1). The action is valid only if the target cell is inside the grid and is not `#`. |
| **Initial state s₀** | The cell marked `S`: `(1, 1)` |
| **Goal G** | `{(7, 15)}`, the cell marked `G` |
| **Cost c** | `c(s, a, s') = 1` for every move, so path cost = number of moves |

**(a) What information specifies a state?** Only the robot's position `(row, col)`. The map is fixed, so it is part of the problem, not the state.

**(b) What makes an action invalid?** The action would move the robot into an obstacle `#` or off the grid.

**(c) Is the problem deterministic?** Yes. Each action has exactly one outcome, and the environment is static and fully observable.

**(d) What is a solution?** A sequence of actions that takes s₀ to G, with every intermediate state free. An *optimal* solution has minimum total cost (fewest moves).

## Task 1: Agent Design (written before prompting the LLM)

1. **State representation:** a tuple `(row, col)`. Tuples are hashable, so they can go in sets and dicts.
2. **Warehouse representation:** a list of strings, indexed as `grid[r][c]`. `S` and `G` are located by scanning the grid.
3. **Valid actions:** `actions(s)` applies each of the four offsets and keeps those whose target cell is in bounds and not `#`.
4. **Goal recognition:** `is_goal(s)` returns `s == goal`. For A*, it is checked when a state is **popped** (expanded), not when it is generated. Only then is optimality guaranteed.
5. **Frontier contents:** a min-heap of `(f, h, tie, state)`, plus two dicts: `g_cost[state]` and `parent[state] = (previous state, action)`.
6. **Path reconstruction:** follow `parent` pointers from the goal back to the start, then reverse.

**Reported on termination:** whether a solution was found, the path, the path length (moves), and the number of states expanded.

**Heuristic:** Manhattan distance `h(n) = |r − r_G| + |c − c_G|`. It is admissible because each move changes `r` or `c` by exactly 1, so at least `h(n)` moves remain. It is consistent because one move changes `h` by at most 1, which equals the step cost.
