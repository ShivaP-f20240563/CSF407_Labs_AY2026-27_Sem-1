# LLM Prompt Log (Agents Lab)

## Prompt 1: initial specification

> Write a well-documented Python program implementing a goal-based agent for the warehouse
> navigation problem shown above (map pasted). The program should
> - represent the warehouse as a two-dimensional grid;
> - determine a collision-free path from S to G;
> - avoid all obstacles;
> - print either the path found or a suitable message if no path exists;
> - explain the search algorithm that has been chosen and why it is appropriate.
> Structure it as an explicit goal-based agent (state, goal, model, planner) with a separate
> Environment class.

**Outcome:** the program ran and used BFS. Review showed that it assumed equal-length rows and did
not validate `S`/`G`.

## Prompt 2: robustness fixes

> The program assumes all map rows have the same length and does not check that S and G each
> appear exactly once. Pad short rows with walls and raise a ValueError for invalid maps. Also
> print the number of states expanded and draw the path on the map.

## Prompt 3: tests

> Write unittest tests: the lab map (verify validity and that the length equals an independent BFS
> oracle), an adjacent goal, an unreachable goal, a map with two routes of different lengths, and a
> larger open map.

## What I changed or verified myself

- Chose BFS and the validation criteria **before** prompting (see `DESIGN.md`).
- Checked by hand that 20 moves is optimal: Manhattan distance 18 plus 2 for the detour at column 6.
- Kept the oracle in the tests independent of the agent code.
