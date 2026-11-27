# LLM Prompt Log (Logic Lab)

LLM used: Claude. Every generated piece was reviewed, run and tested before it was kept.

## Prompt 1: planner (Task 2)

The lab's specification prompt was used, with the warehouse problem from `DESIGN.md` pasted in:

> I want to implement a simple planning agent in Python. Represent a state as a set of logical
> propositions. Each action should contain: a name; positive preconditions; negative preconditions;
> positive effects; negative effects. An action is applicable if all of its preconditions are
> satisfied by the current state. When an action is applied: 1. remove its negative effects from the
> state; 2. add its positive effects to the state. Use breadth-first search to find a sequence of
> actions that achieves a specified goal. The program should also: detect when no plan exists; print
> the resulting sequence of actions; print the states reached after each action. Explain the
> implementation and identify any assumptions you make.

**Stated assumptions:** closed world (absent = false), ground (pre-instantiated) actions, unit costs,
goal is a conjunction of positive literals, finite state space.

## Prompt 2: independent validator

> Add a function `validate(initial, plan_names, actions, goal)` that re-executes a plan given only as
> action names, without using the search code, and reports the first action that is unknown or
> inapplicable together with its unsatisfied preconditions, or reports if the goal is not reached.

## Prompt 3: tests

> Write unittest tests for: the warehouse (plan valid and shortest), the warehouse with PickUp removed
> (no plan), extra robot-only actions (robot at C must not count as package at C), and the
> locked-door Tasks 6(b)–(d).

## Task 7 prompt (locked door, BFS solution not shown)

> Initial state I = {At(A), KeyAt(A)}. Goal G = {At(C)}. Actions (preconditions → effects):
> Move(A,B): At(A) → −At(A), +At(B); Move(B,A): At(B) → −At(B), +At(A);
> PickKey: At(A), KeyAt(A) → −KeyAt(A), +HasKey; Unlock: At(B), HasKey → +Unlocked;
> Move(B,C): At(B), Unlocked → −At(B), +At(C); Move(C,B): At(C), Unlocked → −At(C), +At(B).
> A proposition not in the state is false. Give only a sequence of these six actions that reaches G.

Results are in `REPORT.md` (Task 7).

## What I changed or verified myself

- Wrote the (I, A, G) specification and the hand plans **before** prompting (`DESIGN.md`).
- Caught that the handout's hinted plan (`Move(A,B)`, `PickUp(Package,B)`, …) is invalid, and added a
  test that the validator rejects it at step 2.
- **Checked in the generated BFS:** states are marked visited when they are *generated*, not when
  they are expanded. Otherwise duplicates are queued: `Unlock` stays applicable after unlocking and
  yields the same state. The goal test is also done at generation. This is still shortest under unit
  costs, because BFS generates states in order of depth.
- Generalised `PickUp`/`Drop` to every location. Otherwise the planner could never consider a wrong
  pick-up location, and Test C would be weaker.
- Kept `validate()` separate from `bfs_plan()`: the tests compare the BFS state trace with the
  validator's trace, so the search does not grade itself.
