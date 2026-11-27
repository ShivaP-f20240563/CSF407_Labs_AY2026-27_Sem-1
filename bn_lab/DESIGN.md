# BN Lab: Model Specification (before asking the LLM)

## Question 1: Why the autoregressive decomposition is useful for generation

The chain rule writes the joint distribution exactly (no approximation) as a product of
**next-token conditionals**:
P(x₁,…,x_T) = P(x₁) ∏ₜ P(xₜ | x₁,…,xₜ₋₁).

- **Generation becomes sequential sampling.** Sample x₁, then x₂ given x₁, and so on. We never have to
  enumerate or sample from the joint over all |V|^T sequences.
- **Each factor is a "predict the next word" problem.** Such a problem can be estimated from data
  (counts here, a neural network in an LLM) and evaluated left to right.
- **Variable length comes for free.** Generation stops when `<END>` is sampled.
- **The probability of any sentence is computable** as a product of the factors (or a sum of logs).

## Question 2: Independence assumption of X₁ → X₂ → ⋯ → X_T

Each word is conditionally independent of all earlier words given the immediately preceding word:

  P(Xₜ | X₁,…,Xₜ₋₁) = P(Xₜ | Xₜ₋₁),  equivalently  Xₜ ⊥ {X₁,…,Xₜ₋₂} | Xₜ₋₁.

This is the first-order Markov property. The joint factorises as P(X₁) ∏ₜ P(Xₜ | Xₜ₋₁).

## Variables, dependencies, distribution, computation

| Item | Specification |
|---|---|
| Variables | Xₜ ∈ V = 9 words ∪ {`<START>`, `<END>`}, lower-cased, whitespace tokens |
| Dependencies | First order: Xₜ₋₁ → Xₜ. Second order: (Xₜ₋₂, Xₜ₋₁) → Xₜ |
| Distribution | P(wⱼ \| wᵢ) = C(wᵢ,wⱼ) / Σₖ C(wᵢ,wₖ) (maximum likelihood by counting) |
| P(X₁) | Treated as P(X₁ \| `<START>`), so `<START>` is the root and conditions X₁ |
| Program | Count transitions, normalise, show a CPT row, take the argmax, sample until `<END>` |
| Invariant | Σᵥ P(v \| w) = 1 for every observed context w |

## Question 3: Hand-computed CPT P(next | current)

Counts from the 6 training sentences, with `<START>`/`<END>` added (12 occurrences of `the`):

| Current | Next-word distribution | Row total |
|---|---|---|
| `<START>` | the 6/6 = 1 | 6 |
| `the` | cat 3/12 = 0.25, dog 3/12 = 0.25, mat 2/12 ≈ 0.167, rug 2/12 ≈ 0.167, park 2/12 ≈ 0.167 | 12 |
| `cat` | sat 2/3 ≈ 0.667, ran 1/3 ≈ 0.333 | 3 |
| `dog` | sat 2/3 ≈ 0.667, ran 1/3 ≈ 0.333 | 3 |
| `sat` | on 4/4 = 1 | 4 |
| `ran` | to 2/2 = 1 | 2 |
| `on` | the 1 | 4 |
| `to` | the 1 | 2 |
| `mat`, `rug`, `park` | `<END>` 1 | 2 each |

Note that P(cat | the) = 3/12, not 3/5 as in the handout's toy example. `the` is followed by 12 words in
this corpus, because it also precedes mat, rug and park.

**Zero-probability transitions (examples).** Every pair not in the table is zero, for instance:

- P(ran | sat) = 0 and P(the | sat) = 0: `sat` is only ever followed by `on`.
- P(park | on) = 0, P(mat | to) = 0: the bigram model never sees `on` → `park` directly. "on the park"
  still has non-zero probability, because the bigrams route through `the`.
- P(<END> | the) = P(<END> | cat) = 0: sentences never end on these words.
- P(cat | cat) = 0, and every self-transition is zero.
- **Unknown contexts.** A word outside V, such as `bird`, has no CPT row at all. The model must handle
  this explicitly (see Question 7).

Of the 10 × 10 = 100 possible (context, next) cells (contexts: `<START>` + 9 words; next: 9 words +
`<END>`), only **17 are non-zero**. The CPT is very sparse even for this tiny corpus.
