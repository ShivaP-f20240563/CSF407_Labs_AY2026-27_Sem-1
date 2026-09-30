# Agents Lab Report: A Goal-Based Agent with LLM Assistance

**Course:** CS F407 Artificial Intelligence
**Files:** `DESIGN.md` (Tasks 1–2), `warehouse_agent.py` (agent), `test_warehouse_agent.py` (tests), `PROMPTS.md` (LLM log)

## How to run

```bash
python warehouse_agent.py            # solve the lab map
python warehouse_agent.py mymap.txt  # solve any ASCII map
python -m unittest -v                # run the test suite
```

## Result on the lab map

```
Path found: 20 moves, 59 states expanded by BFS
Actions : Right Right Right Down Right Right Right Up Right x12

#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

The vehicle drops into row 2 to get around the shelf at (1, 6), then returns to row 1 and runs
straight to `G`. The Manhattan distance from S to G is 18, and the detour around the wall adds
2 moves, so **20 is the minimum**. The independent BFS oracle in the tests confirms this.

## Test results

All 7 tests pass (`python -m unittest -v`):

| Test | Checks | Result |
|---|---|---|
| Lab map | Path is valid (no walls, unit moves) and shortest (matches oracle) | Pass |
| Adjacent goal `#SG#` | Returns the single action `Right` | Pass |
| Vertical neighbour | Returns `Down` | Pass |
| Unreachable goal | Reports failure, no infinite loop | Pass |
| Two routes (4 vs 8 moves) | Picks the 4-move route | Pass |
| 42 × 22 open map | Still valid and shortest | Pass |
| Map without `G` | Raises `ValueError` | Pass |

## Task 3: Questions

**1. Did the LLM generate a working program on the first attempt?**
Yes, for the lab map: the first version found a valid 20-move path. Code review still found two
robustness gaps, which were added in a follow-up commit:

- It assumed every row had the same length, so ragged maps were unsafe. Rows are now padded with `#`.
- It did not check that the map contained exactly one `S` and one `G`. It now raises a clear
  `ValueError`.

**2. How can the prompt be improved?**
State the edge cases and the output format explicitly. For example: "handle maps with no path,
malformed maps and ragged rows"; "return both the list of actions and the list of cells"; "separate
the environment from the agent into their own classes"; "include unit tests with an independent
oracle". A precise specification of *what counts as correct* gives better code than asking for
"a program".

**3. Which search algorithm did the LLM choose?**
Breadth-first search, with a FIFO queue, a visited set, and parent pointers for path
reconstruction.

**4. Why did the LLM select this algorithm?**
The problem has uniform step costs, a small discrete state space and a known goal. Those are the
textbook conditions under which BFS is complete and optimal. It is also the simplest correct choice.
A* with the Manhattan heuristic would give the same path length while expanding fewer states. That
matters only on larger maps; it is examined in the Search lab.

## Reflection

- **What the LLM did well:** it produced boilerplate quickly (grid parsing, the BFS loop, path
  reconstruction) and explained why BFS is optimal here.
- **What still needed a human:** specifying the agent architecture, deciding what "correct" means,
  and writing an independent oracle so the program's output could be *verified* rather than trusted.
- **Limitations:** LLM code is a hypothesis. It looked right on the one example map but was not
  robust until it was tested against deliberately chosen edge cases.
