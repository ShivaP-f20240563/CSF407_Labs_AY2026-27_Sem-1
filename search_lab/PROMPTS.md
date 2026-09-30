# LLM Prompt Log (Search Lab)

## Prompt 1: A* generation (based on my Task 1 design)

> I am implementing a simple goal-based search agent in Python. The environment is a grid represented
> by an ASCII map. The agent starts at S and must reach G. The symbols # represent obstacles and .
> represents free cells. The agent can move up, down, left, or right, and every movement has cost 1.
> Implement A* search. Use Manhattan distance as the heuristic h(n) = |x − xG| + |y − yG|.
> The program should: represent grid positions as (row, col) tuples; keep a GridProblem class with
> actions(), result(), is_goal() and cost(); maintain a heapq frontier; calculate g(n), h(n) and f(n)
> explicitly; avoid repeatedly expanding the same state (closed set); do the goal test when a node is
> expanded; reconstruct the path; report the path, its length and the number of states expanded.
> Keep the implementation simple and explain the main components of the code.

**Accepted:** the A* loop, the closed set, parent pointers and the `GridProblem` interface.

## Prompt 2: BFS and pluggable heuristics

> Add a BFS function with the same reporting, and make the A* heuristic a parameter. Provide
> h = 0, Euclidean distance, and a helper that multiplies any heuristic by a constant.

## Prompt 3: explanation

> Why is Manhattan distance an appropriate heuristic for this warehouse when the robot can move only
> horizontally and vertically? Is it admissible? Consistent?

## Prompt 4: tie-breaking fix (after my experiment showed A* = BFS on an open map)

> On an open 20×9 map, A* with Manhattan distance expands 117 states, the same as BFS. Why? Change
> the priority so that ties in f are broken in favour of smaller h, without affecting optimality.

## Human decisions and verification

- The problem formulation, the tests and the experiment design were mine, written before prompting.
- Every path length was checked against BFS. The trap map for the inflated heuristic was found by a
  small random search and checked against BFS.
- The heuristic conclusions come from the measured numbers in `results.md`, not from the LLM's answer.
