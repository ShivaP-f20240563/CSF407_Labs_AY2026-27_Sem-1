# Search Lab Report: A* with an LLM as Engineering Assistant

**Course:** CS F407 Artificial Intelligence

| File | Contents |
|---|---|
| `DESIGN.md` | Task 0 (problem formulation) and Task 1 (agent design) |
| `search_agent.py` | A*, BFS, heuristics, CLI |
| `test_search_agent.py` | Task 3 tests (7 unit tests) |
| `experiments.py` → `results.md` | Task 5 and Task 6 tables (reproducible) |
| `PROMPTS.md` | LLM prompts and what was accepted or changed |

```bash
python search_agent.py          # A* and BFS on the lab map
python -m unittest -v           # Task 3 tests
python experiments.py           # regenerates results.md
```

---

## Task 2: A* Generated with the LLM

The prompt was the example prompt from the lab sheet, extended with the design from Task 1 (see
`PROMPTS.md`). Result on the lab warehouse:

```
A*[manhattan]: found=True  length=40  expanded=64
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

## Task 3: Testing

| Test | Map | Expected | Observed |
|---|---|---|---|
| 1. Original warehouse | lab map | valid shortest path | Found, length 40 (equals BFS), 64 expanded |
| 2. Trivial | `#SG##` | 1 move | `['Right']` |
| 3. No solution | lab-sheet map (G walled off) | failure, terminates | `found=False` for both A* and BFS |
| 4a. Alternative paths, equal | two routes of length 9 | length 9 | 9 |
| 4b. Alternative paths, unequal | routes of 6 and 10 | 6 | 6 |
| Extra | h = 0 / Euclidean | same optimal length | 40 |
| Extra | map without `G` | error | `ValueError` |

Every returned path is also checked step by step: it starts at S, ends at G, each move is a unit
move, and it never enters a `#` cell.

## Task 4: Where the A* Concepts Appear in `search_agent.py`

| Concept | Location |
|---|---|
| State | `State = Tuple[int, int]`, a `(row, col)` tuple |
| Action | `ACTIONS` dict; `GridProblem.actions(s)` returns the valid ones |
| Transition | `GridProblem.result(s, a)` (via `_move`) |
| Goal test | `GridProblem.is_goal(s)`, called right after a state is popped in `astar` |
| g(n) | `g2 = g_cost[s] + problem.cost(...)`, stored in the `g_cost` dict |
| h(n) | `h2 = h(s2, goal)` (the heuristic function is passed in) |
| f(n) | `f2 = g2 + h2`, the first key of each heap entry |
| Frontier | `frontier`, a binary heap (`heapq`) of `(f, h, tie, state)` |
| Visited states | `closed` set; stale heap entries are skipped with `if s in closed: continue` |
| Path reconstruction | `_reconstruct(parent, goal)` follows parent pointers, then reverses |

- **(a) Frontier structure:** a binary min-heap (priority queue) from `heapq`.
- **(b) Selecting the next state:** `heappop` returns the entry with the smallest `f`. Ties go to the smaller `h`, then to insertion order.
- **(c) Where the heuristic is computed:** when a successor is generated (`h2 = h(s2, goal)`), and once for the start state.
- **(d) Is f = g + h calculated explicitly?** Yes: `f2 = g2 + h2`.
- **(e) Avoiding repeated exploration:** the `closed` set means a state is never expanded twice. A successor is pushed only if its new `g` beats the best known `g`.

## Task 5: BFS vs A*

**Lab warehouse:**

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | Yes | Yes |
| Path length | 40 | 40 |
| States expanded | 64 | 64 |

**Extra: open warehouse floor** (map in `results.md`):

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | Yes | Yes |
| Path length | 23 | 23 |
| States expanded | 117 | 44 |

**(a) Did both find a solution?** Yes.

**(b) Same path length?** Yes. Both are optimal: BFS because costs are uniform, A* because Manhattan distance is admissible.

**(c) Which expanded fewer states?** On the lab map, neither. Both expanded all 64 free cells. On the open floor, A* expanded 44 against BFS's 117.

**(d) Why can A* expand fewer states?** It uses information about *where the goal is*. The lab map is a
winding maze whose only route leads away from G for most of its length, so Manhattan distance is a
poor estimate there. Nearly every cell ends up with f ≤ 40, so A* has to look at everything, just
like BFS. On an open floor the heuristic is accurate and A* heads almost straight for G. The benefit
of A* depends on the heuristic's quality relative to the map, not on the algorithm being "smarter".

## Task 6: Heuristic Investigation

The LLM's explanation of why Manhattan distance suits this problem: with only 4-directional unit moves, the
fewest possible moves to G, ignoring walls, is exactly `|Δr| + |Δc|`. Walls can only add moves, so
Manhattan distance never overestimates (it is admissible). It is also the *tightest* such bound
without knowledge of the walls.

Full tables are in `results.md`. Summary:

| Heuristic | Lab map: length / expanded | Open floor | Trap map | Admissible? |
|---|---|---|---|---|
| Manhattan | 40 / 64 | 23 / 44 | 16 / 49 | Yes |
| h = 0 | 40 / 64 | 23 / 117 | 16 / 55 | Yes (A* becomes uniform-cost search, i.e. BFS order) |
| Euclidean | 40 / 64 | 23 / 96 | 16 / 48 | Yes, but looser than Manhattan on a 4-connected grid |
| 2 × Manhattan | 40 / 64 | **27** / 30 | **24** / 36 | **No** |
| 5 × Manhattan (extra) | **48** / 64 | **27** / 29 | **24** / 28 | **No** |

**Observations:**

1. **h = 0:** A* reduces to uniform-cost search. It is still optimal, but it expands the same number of states as BFS because it has no goal information.
2. **Euclidean:** still admissible, since a straight line is never longer than the Manhattan route, so paths stay optimal. But it *underestimates* more, so it guides less and expands more states than Manhattan on the open floor (96 against 44).
3. **2 × Manhattan:** it overestimates, so it is *not admissible*. The search becomes greedier and expands far fewer states (30 against 44, and 36 against 49), but it returned **non-optimal paths**: 27 instead of 23, and 24 instead of 16. On the lab map, 2× happened to stay optimal. 5× did not (48 instead of 40), which shows that inadmissibility *removes the guarantee*; it does not guarantee failure.

**Conclusion:** the more accurate an admissible heuristic is (closer to h\*), the fewer states A*
expands while staying optimal. Once `h > h*`, A* trades optimality for speed. It starts behaving
like greedy best-first search.

## Task 7: Evaluating the LLM-Generated Agent

1. **Correct immediately:** grid parsing, the priority-queue A* loop, the closed set, path reconstruction, and the BFS variant.
2. **Bugs or design problems:** the first version broke ties among equal `f` values in FIFO order. On the open floor it therefore expanded **117 states, exactly as many as BFS**, even though Manhattan distance is a near-perfect heuristic there. In an open grid, every cell inside the S–G rectangle has the same `f`, and FIFO explores them level by level. Breaking ties on smaller `h` brought this down to 44, with the path still optimal.
3. **How it was found:** by comparing expansion counts on a *second* map where A* was expected to win. The lab map alone could not show it, because both algorithms expand everything there.
4. **Unfamiliar terminology or structures:** `heapq` tuple ordering (tuples compare element by element, which is what makes the `(f, h, tie, state)` trick work), and "lazy deletion" of stale heap entries.
5. **Modifications:** added the `h` tie-breaker, the pluggable heuristic parameter, the `SearchResult` dataclass, and the `experiments.py` harness.
6. **Most useful tests:** the no-solution map (termination), the unequal-routes map (optimality), and comparing against BFS as an independent oracle.
7. **Trust without testing?** No. The first version gave correct paths but hid an efficiency flaw, and inflated heuristics give *plausible-looking* but non-optimal paths.
8. **New understanding:** A*'s benefit depends entirely on how the heuristic interacts with the map; the goal test must happen at expansion time; tie-breaking matters in practice.

**Who contributed what:**

| | |
|---|---|
| Designed by me | Problem formulation, state and frontier design, test cases, experiment plan |
| Suggested by the LLM | Code skeleton, `heapq` usage, lazy deletion |
| Accepted | Core A* and BFS loops |
| Changed | Tie-breaking, heuristic plumbing, reporting |
| Tested | Everything in Task 3, plus BFS cross-checks for every map |

## Final Reflection

**1. Why formulate the problem first?** The formulation fixes what a state is, what counts as a
valid move and what the goal is. Without it there is no way to judge whether generated code is
correct. For example, if the state accidentally included the path so far, the closed set would never
prune anything. Specifying the problem first turns code review into checking the code against a spec.

**2. In what sense is A* informed?** Besides the cost already paid (`g`), it uses problem-specific
knowledge of the goal (`h`) to rank frontier nodes. BFS orders nodes only by depth. A* orders them by
an estimate of total solution cost through each node. This lets it ignore regions that cannot lead to
a cheaper solution.

**3. Why does the heuristic matter?** It controls both the *efficiency* and the *correctness* of A*.
An admissible heuristic guarantees optimality, and a tighter one expands fewer states (open floor:
h = 0 gave 117 expansions, Euclidean 96, Manhattan 44). An inadmissible heuristic expands even fewer
states but can return longer paths (27 instead of 23, and 24 instead of 16).

**4. What did the LLM contribute?** It turned a clear design into working code quickly and explained
library details such as `heapq` ordering. It sped up the boilerplate; it did not do the reasoning
about what to test.

**5. Risks of accepting code without testing:** the tie-breaking flaw would have gone unnoticed, since
the code still found correct paths. With an inflated heuristic, the robot would take longer routes and
nobody would know. In a real warehouse that means wasted energy, or worse: code that looks correct
but is only validated on one example.
