# Logic Lab: Problem Specification and Hand Plan

## Task 0: The warehouse planning problem (I, A, G)

Locations `A`, `B`, `C` with passages `A ↔ B` and `B ↔ C` only (no direct `A ↔ C`).

| Item | Value |
|---|---|
| **(a) Initial state I** | `{At(Robot,A), At(Package,A)}` |
| **(b) Goal G** | `{At(Package,C)}` (nothing is said about the robot) |
| **(c) Actions** | `Move(A,B)`, `Move(B,A)`, `Move(B,C)`, `Move(C,B)`, `PickUp(Package,x)`, `Drop(Package,x)` for `x ∈ {A,B,C}` |

Closed-world assumption: a proposition not in the state is false.

### (d) Preconditions and effects

| Action | Preconditions | Delete (¬) | Add |
|---|---|---|---|
| `Move(x,y)`, `(x,y)` connected | `At(Robot,x)` | `At(Robot,x)` | `At(Robot,y)` |
| `PickUp(Package,x)` | `At(Robot,x)`, `At(Package,x)` | `At(Package,x)` | `Holding(Package)` |
| `Drop(Package,x)` | `At(Robot,x)`, `Holding(Package)` | `Holding(Package)` | `At(Package,x)` |

While the package is held it has no `At(Package,·)` fact; its location is implied by the robot's.

### Which actions are applicable in I?

- **`PickUp(Package,A)` is applicable.** Both preconditions, `At(Robot,A)` and `At(Package,A)`, are in I,
  so `I ⊨ Pre(PickUp(Package,A))`.
- **`Drop(Package,C)` is not applicable.** Neither `At(Robot,C)` nor `Holding(Package)` is in I. Under the
  closed-world assumption both are false, so the precondition conjunction fails.
- The only other applicable action in I is `Move(A,B)`.

Being in the action list only means an action *exists*; whether it can be executed is a logical
question, `S ⊨ Pre(a)`, asked again in every state.

## Task 1: Hand-constructed plan

The handout's hint (`Move(A,B)`, `PickUp(Package,B)`, …) is **not valid**: after `Move(A,B)` the
package is still at `A`, so `At(Package,B)` is false and `PickUp(Package,B)` cannot be applied.
The robot has to pick the package up before leaving `A`:

| State | Action that led here | Facts |
|---|---|---|
| S0 | (initial) | `At(Robot,A)`, `At(Package,A)` |
| S1 | `PickUp(Package,A)` | `At(Robot,A)`, `Holding(Package)` |
| S2 | `Move(A,B)` | `At(Robot,B)`, `Holding(Package)` |
| S3 | `Move(B,C)` | `At(Robot,C)`, `Holding(Package)` |
| S4 | `Drop(Package,C)` | `At(Robot,C)`, `At(Package,C)` |

`S4 ⊨ G`. Four actions is the minimum: the robot needs two moves to reach `C` (no `A ↔ C` passage),
one pick-up and one drop.

## Task 6(a): Locked-door domain, hand plan

`I = {At(A), KeyAt(A)}`, `G = {At(C)}`. `B ↔ C` needs `Unlocked`, which needs `HasKey` at `B`.

| State | Action | Facts |
|---|---|---|
| S0 | (initial) | `At(A)`, `KeyAt(A)` |
| S1 | `PickKey` | `At(A)`, `HasKey` |
| S2 | `Move(A,B)` | `At(B)`, `HasKey` |
| S3 | `Unlock` | `At(B)`, `HasKey`, `Unlocked` |
| S4 | `Move(B,C)` | `At(C)`, `HasKey`, `Unlocked` |

Shortest: reaching `C` needs `Move(B,C)`, which needs `Unlocked`, which needs `Unlock`, which needs
`HasKey`, which only `PickKey` provides. So all four actions are necessary.

## Program design (specification given to the LLM)

| Concept | Representation in `planner.py` |
|---|---|
| State | `frozenset` of proposition strings, e.g. `"At(Robot,A)"` (hashable, so it can go in a visited set) |
| Action | `Action(name, pre_pos, pre_neg, add, delete)` |
| Applicability (logic) | `pre_pos ⊆ S` and `pre_neg ∩ S = ∅` |
| Transition | `S' = (S − delete) ∪ add` |
| Goal test | `goal ⊆ S` |
| Search | BFS over states, visited set, parent pointers, goal test on generation |
| Independent check | `validate(I, plan, actions, G)` re-executes a plan and reports the first failing precondition |

BFS is chosen because every action has unit cost, so BFS returns a shortest plan, and the state space
is finite, so BFS terminates with "No plan found" when the goal is unreachable.
