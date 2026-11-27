# Logic Lab Report: Logical Reasoning for Planning

**Course:** CS F407 Artificial Intelligence
**Files:** `DESIGN.md` (spec, hand plans), `planner.py` (planner + validator), `test_planner.py`,
`planner.pl` (Prolog verifier), `results.txt`, `prolog_results.txt`, `PROMPTS.md` (LLM log)

## How to run

```bash
python planner.py                                       # all domains -> results.txt
python -m unittest -v                                   # 10 tests
python planner.py door "PickKey, Move(A,B), Unlock, Move(B,C)"   # validate any plan
swipl -q -s planner.pl -g run_queries -t halt           # Prolog queries
```

## Task 0 and Task 1

See `DESIGN.md`. `PickUp(Package,A)` is applicable in I, because both of its preconditions hold.
`Drop(Package,C)` is not, because `At(Robot,C)` and `Holding(Package)` are both false. The hand plan is
`PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C)`. The handout's hinted ordering, which picks
the package up at B, fails at step 2.

## Task 2: Where the specification appears in the code

| Idea | Code |
|---|---|
| Preconditions | `Action.applicable`: `pre_pos ⊆ S and pre_neg ∩ S = ∅` |
| Effects | `Action.apply`: `(S − delete) ∪ add` (delete first, then add) |
| Goal | `satisfies(s, goal)`: `goal ⊆ s`, tested when a state is generated |
| BFS | `bfs_plan`: FIFO `deque`, `parent` dict as visited set and back-pointers |

## Task 3: Test results (all 10 pass)

| Test | Initial state | Goal | Plan found? | Plan | Valid? |
|---|---|---|---|---|---|
| A: solvable | `At(Robot,A), At(Package,A)` | `At(Package,C)` | Yes (4) | PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C) | Yes. The validator re-executes it step by step, and its trace equals the BFS trace |
| B: no PickUp | same | `At(Package,C)` | **No plan found** | — | Correct: the only way to obtain `Holding(Package)` is removed |
| C: irrelevant actions (`Move(A,C)` shortcut, no-op `Wait`) | same | `At(Package,C)` | Yes (3) | PickUp(Package,A), Move(A,C), Drop(Package,C) | Yes. `[Move(A,C)]` alone is rejected: "goal not reached; missing At(Package,C)" |
| Hint plan | same | `At(Package,C)` | — | Move(A,B), PickUp(Package,B), … | **Invalid** at step 2: `At(Package,B)` unsatisfied |

Test C shows that the goal test checks the package's location, not the robot's. The robot reaching C
is not enough, and `Wait` never appears in the plan, because it produces no new state.

## Task 4: Logic and search

```
Current state
     ↓
Check action preconditions        (logic: S ⊨ Pre(a)?)
     ↓
Select an applicable action       ← the missing step
     ↓
Generate successor state          S' = (S − Del(a)) ∪ Add(a)
     ↓
Search over alternatives          (BFS frontier, visited set)
     ↓
Goal?                             (logic: S' ⊨ G?)
```

Logic is used twice: to decide which actions are possible in a state, and to decide whether a state
satisfies the goal. Search does not reason about *why* an action is possible. It chooses which
applicable action and which state to explore next, avoids revisiting states, and guarantees that the
first goal state found is reached by a shortest action sequence. Logic determines what is possible;
search determines what to try.

## Task 5: Can the LLM verify its own plan?

Asked to justify each step, the LLM produced a correct-looking explanation for the 4-step warehouse
plan, which matched `results.txt`. **Trust (b), the executed state transitions.** The explanation is
itself generated text. It can say "the robot is holding the package" without that being a computed
fact, and the same fluent style was used for the invalid hint plan. The executed transitions are a
mechanical check against the action definitions: each precondition is a subset test on an explicit
state. A generated explanation is not the same as an independent verification.

## Task 6: Locked door

| Part | Result |
|---|---|
| (a) Hand plan | PickKey, Move(A,B), Unlock, Move(B,C): see `DESIGN.md` for the state table |
| (b) BFS | Same 4-step plan, 5 states expanded. The trace is in `results.txt` |
| (c) No PickKey | **No plan found**. `HasKey` is only added by `PickKey`, `Unlock` needs `HasKey`, and `Move(B,C)` needs `Unlocked`, so `At(C)` is unreachable. Only {At(A),KeyAt(A)} and {At(B),KeyAt(A)} are reachable |
| (d) Validate `Move(A,B), Unlock, Move(B,C)` | **Invalid. First invalid action: step 2, `Unlock`; unsatisfied precondition `HasKey`** |

## Task 7: LLM plans across two domains

| Plan checked | Source | Valid? | Reaches goal? | Same length as BFS? |
|---|---|---|---|---|
| PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C) | LLM, warehouse | Yes | Yes | Yes (4) |
| PickKey, Move(A,B), Unlock, Move(B,C) | LLM, locked door (prompt in `PROMPTS.md`) | Yes | Yes | Yes (4) |
| Move(A,B), Unlock, Move(B,C) | Typical "skip the key" proposal (Task 6d) | **No**: step 2 Unlock, `HasKey` unsatisfied | No | — |

The validator's first error ("step 2: Unlock not applicable; unsatisfied: HasKey") is exactly the
feedback the lab asks us to return to the LLM. Adding `PickKey` first gives the valid 4-step plan.
Any proposal can be checked with `python planner.py door "<plan>"`.

The flawed sequence reads as a plausible story ("go to B, unlock the door, go through"), but it
skips the key. A plausible narrative does not guarantee that preconditions hold. The independent
checker establishes, for every step, that each precondition is literally in the current state, and it
points to the exact failing step and literal. A narrative cannot do either.

## Tasks 8–10: Prolog (`prolog_results.txt`)

```
?- can_move(a,b).                       true
?- can_move(a,c).                       false
?- valid_move(a,b).                     true
?- valid_move(b,c).                     true
?- valid_move(a,c).                     false
?- valid_plan(a,[move(a,b),move(b,c)]). true
?- valid_plan(a,[move(a,c)]).           false
?- reduce_speed.                        true
```

- **8(a)** `can_move(a,b)` succeeds because Prolog unifies the rule head with X=a, Y=b, and the body
  `connected(a,b)` matches a fact.
- **8(b)** `can_move(a,c)` fails because no fact `connected(a,c)` exists and there is no transitive
  rule. Under Prolog's closed-world reading (negation as failure), what cannot be derived is reported
  false. Reachability via B would need a recursive rule.
- **8(c)** `can_move(X,Y) :- connected(X,Y).` is the Horn clause ∀X,Y Connected(X,Y) → CanMove(X,Y).
  The head is the consequent and the body is the antecedent.
- **Task 9 challenge.** `valid_move(a,c)` is false, so the proposed `Move(a,c)` is **not supported** by
  the warehouse knowledge. I also added `valid_plan/2`, which checks that a whole move sequence chains
  from the start location.
- **Task 10.** `reduce_speed` succeeds by backward chaining: to prove `reduce_speed` it proves
  `slippery`, and to prove that it proves `wet_road`, which is a fact. As implications:
  **WetRoad (fact) ⇒ (WetRoad → Slippery) ⇒ Slippery ⇒ (Slippery → ReduceSpeed) ⇒ ReduceSpeed**.

## Reflection questions

1. **Why specify preconditions and effects first?** They are the specification the code must
   implement and the yardstick for checking it. Without them the LLM invents its own action semantics,
   and there is nothing to test against.
2. **An error from skipping precondition checks.** The planner would accept `PickUp(Package,B)`
   while the package is at A, or `Move(B,C)` through a locked door. The plan would look complete but
   could not be executed.
3. **Why a plan that "looks reasonable" may be invalid.** Plausibility is about the story of the plan;
   validity is about every literal in every precondition holding in the exact state at that step. The
   hint plan and the first locked-door plan both read well and both fail at step 2.
4. **What the LLM contributed.** The implementation boilerplate (dataclass, BFS with back-pointers,
   printing), the validator and test scaffolding, and a list of its assumptions.
5. **What I verified independently.** The hand plans and their optimality, that the BFS trace equals
   the validator's trace, the no-plan cases, the goal-test semantics in Test C, and the change that
   moved the BFS goal test to generation time (`PROMPTS.md`).
6. **Where logic is used.** Precondition entailment `S ⊨ Pre(a)`, goal entailment `S ⊨ G`, the
   closed-world assumption, and the Prolog rule inference.
7. **Relation to search.** It is the same graph search as the search lab. The nodes are now sets of
   propositions instead of grid cells, and the successor function is derived logically from action
   schemas. BFS stays complete and optimal for unit costs, and A* with a planning heuristic would be the
   next step.

### Prolog reflection

1. **Fact vs rule.** A fact is an unconditional assertion (`connected(a,b).`). A rule is a conditional
   one, `Head :- Body`, which holds when the body can be proved.
2. **A query as entailment.** `?- G.` asks whether G follows from the knowledge base (KB ⊨ G, via SLD
   resolution). "true" means a proof was found; "false" means none could be found under the closed
   world.
3. **Why a Prolog verifier.** It checks the plan against a separately written declarative model, so a
   bug in the Python transition code is not repeated in the checker.
4. **Why an independent verifier helps with LLM plans.** The LLM's plan and its explanation share
   the same failure modes. A separate logical system gives a check that does not depend on the
   generator being right.

**Takeaway:** Logic + Search = Planning, and the pipeline is Understand → Specify → Generate → Execute → Verify.
