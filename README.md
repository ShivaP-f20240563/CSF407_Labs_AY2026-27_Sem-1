# CS F407 Artificial Intelligence: Lab Assignments (AY 2026-27, Sem 1)

Solutions to the CS F407 laboratory exercises, BITS Pilani. Each lab lives on **its own branch**
with an incremental commit history: design first, then implementation, tests, experiments and
report. This mirrors the "specify → generate → test → validate" workflow the labs ask for.

| Lab | Branch | Topic | Main files |
|---|---|---|---|
| 1 | [`agents-lab`](../../tree/agents-lab/agents_lab) | Goal-based agent for warehouse navigation (BFS) | `warehouse_agent.py`, `REPORT.md` |
| 2 | [`search-lab`](../../tree/search-lab/search_lab) | A* vs BFS, heuristic admissibility experiments | `search_agent.py`, `experiments.py`, `REPORT.md` |
| 3 | [`neural-models-lab`](../../tree/neural-models-lab/neural_models_lab) | XOR with a 2-2-1 network: backprop, symmetry, activations, softmax | `xor_lab.py`, `REPORT.md` |
| 4 | [`logic-lab`](../../tree/logic-lab/logic_lab) | STRIPS planning: logic + BFS, plan validator, Prolog verifier | `planner.py`, `planner.pl`, `REPORT.md` |
| 5 | [`bn-lab`](../../tree/bn-lab/bn_lab) | Bayesian networks as autoregressive LMs: 1st/2nd-order n-grams | `ngram_lm.py`, `experiments.py`, `REPORT.md` |

```bash
git clone https://github.com/ShivaP-f20240563/CSF407_Labs_AY2026-27_Sem-1.git
cd CSF407_Labs_AY2026-27_Sem-1
git checkout agents-lab          # or search-lab / neural-models-lab / logic-lab / bn-lab
```

## Lab 1: Goal-Based Agent (`agents-lab`)

An autonomous warehouse vehicle must find a collision-free path from `S` to `G` on a grid.

- An explicit agent architecture (state, goal, world model, planner) with a separate environment class.
- **BFS** planner: complete, and optimal for unit step costs. It finds the **20-move** shortest path.
- 7 unit tests against an independent BFS oracle: no-path, adjacent goal, multiple routes, a large map, invalid maps.

```bash
cd agents_lab && python warehouse_agent.py && python -m unittest -v
```

## Lab 2: Search and A* (`search-lab`)

The warehouse is formulated as a search problem P = (S, A, T, s₀, G, c). A* is compared with BFS, and
the heuristic is varied.

- A* (priority queue on f = g + h, closed set, goal test on expansion) and BFS, both reporting path, length and states expanded.
- Task 3 test suite: original map, trivial map, unreachable goal, alternative paths.
- Heuristic study: Manhattan, h = 0, Euclidean, 2× and 5× Manhattan, on three maps.
  - Admissible heuristics always stay optimal.
  - Inflated heuristics expand fewer states but return **longer paths** (27 instead of 23, 24 instead of 16).
- Engineering finding: FIFO tie-breaking made A* expand as many states as BFS on open floors (117).
  Breaking ties on smaller h cut this to 44.

```bash
cd search_lab && python search_agent.py && python -m unittest -v && python experiments.py
```

## Lab 3: Neural Models (`neural-models-lab`)

A sensor-disagreement detector (XOR) is used to study representation and learning.

| Experiment | Key result |
|---|---|
| Linear baseline | Stuck at loss ln 2 = 0.693, p = 0.5 everywhere |
| 2-2-1 tanh network | Loss 0.71 → 1e-4, 4/4 correct |
| Backprop check | Batch gradient = mean of per-example gradients; autograd matches finite difference |
| Symmetry | Identical initialisation keeps both hidden rows identical at every step |
| Activations | Tanh and sigmoid solve it with seed 0; ReLU gets stuck in its linear region |
| Repeated seeds | 2-2-1 succeeds in only 35–80% of seeds depending on optimiser; 4 hidden units reach 90–95% |
| 3-class softmax | Probabilities sum to 1, shift-invariant, and ∂L/∂z = p − y verified numerically |

```bash
pip install -r requirements.txt
cd neural_models_lab && python xor_lab.py
```

## Lab 4: Logical Planning (`logic-lab`)

A warehouse robot must deliver a package from A to C, and a second domain has a locked door and a key.

- States are sets of propositions. An action applies iff S ⊨ Pre(a), and S' = (S − Del) ∪ Add. BFS returns a shortest plan.
- An independent `validate()` re-executes any plan and reports the first unsatisfied precondition.
  - The handout's hinted plan fails at step 2.
  - `Move(A,B), Unlock, Move(B,C)` fails at `Unlock`, because `HasKey` is false.
- 10 tests: solvable (4 steps), no `PickUp` / no `PickKey` gives "No plan found", and irrelevant robot-only actions.
- SWI-Prolog checks moves and move sequences against the warehouse facts.

```bash
cd logic_lab && python planner.py && python -m unittest -v
swipl -q -s planner.pl -g run_queries -t halt
```

## Lab 5: Bayesian Networks and Autoregressive LMs (`bn-lab`)

| Experiment | Key result |
|---|---|
| CPT by counting | 17 non-zero entries of 121, and they match the hand table exactly |
| Invariant tests | Rows sum to 1, counts match an oracle, sampling frequencies match the CPT, triples marginalise to pairs |
| Greedy vs sampling | 1st-order greedy loops (`the cat sat on the cat …`); 2nd-order greedy ends |
| 1st vs 2nd order | 188 vs 6 distinct sentences in 1000 samples; 0 vs 106 unseen contexts; 2nd order only reproduces training data |

```bash
cd bn_lab && python experiments.py && python -m unittest -v
```

## Repository structure

```
main
└── README.md, requirements.txt, .gitignore
agents-lab
└── agents_lab/          DESIGN.md · warehouse_agent.py · test_warehouse_agent.py · PROMPTS.md · REPORT.md
search-lab
└── search_lab/          DESIGN.md · search_agent.py · test_search_agent.py · experiments.py · results.md · PROMPTS.md · REPORT.md
neural-models-lab
└── neural_models_lab/   DESIGN.md · xor_lab.py · results.txt · PROMPTS.md · REPORT.md
logic-lab
└── logic_lab/           DESIGN.md · planner.py · test_planner.py · planner.pl · results.txt · prolog_results.txt · PROMPTS.md · REPORT.md
bn-lab
└── bn_lab/              DESIGN.md · ngram_lm.py · test_ngram_lm.py · experiments.py · results.txt · generated_sentences.txt · PROMPTS.md · REPORT.md
```

Every lab folder contains the same documents:

| File | Purpose |
|---|---|
| `DESIGN.md` | Problem formulation and agent/model design, written *before* generating code |
| `REPORT.md` | Results, answers to every task question, and reflection |
| `PROMPTS.md` | LLM prompts used, and what was accepted, changed and verified (as the labs require) |

## Requirements

- Python 3.9+
- Labs 1, 2, 4 and 5 use only the standard library. Lab 4's optional Prolog part needs SWI-Prolog.
- Lab 3 needs `torch` (CPU is enough) and `numpy`.

## Author

Shiva Pillutla (2024A7PS0563G), CS F407 Artificial Intelligence, BITS Pilani K K Birla Goa Campus.
