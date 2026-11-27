# BN Lab Report: Bayesian Networks and Autoregressive Language Models

**Course:** CS F407 Artificial Intelligence
**Files:** `DESIGN.md` (Q1–3, hand CPT), `ngram_lm.py` (first- and second-order model),
`test_ngram_lm.py` (19 tests), `experiments.py` → `results.txt`, `generated_sentences.txt`,
`PROMPTS.md` (LLM log)

## How to run

```bash
python ngram_lm.py          # CPT rows and argmax for both orders
python -m unittest -v       # 19 invariant tests
python experiments.py       # writes results.txt and generated_sentences.txt (seed 407)
```

Standard library only.

## Questions 1–3

See `DESIGN.md`. In short: the chain rule turns generation into sequential next-token sampling (Q1).
The first-order network assumes Xₜ ⊥ X₁..ₜ₋₂ | Xₜ₋₁ (Q2). The CPT has 17 non-zero entries out of 121
(Q3).

## Questions 4–7: Inspecting the code

| Q | Answer |
|---|---|
| **4. Where are counts stored?** | `self.counts`: a `defaultdict(Counter)` mapping context tuple → next token → count, filled in `__init__` |
| **5. Where is P(Xₜ \| Xₜ₋₁) computed?** | `self.probabilities` in `__init__`: each row is divided by its row total, `c / sum(nxt.values())`. `distribution(ctx)` looks a row up |
| **6. How is the next word chosen?** | Both are supported. `mode="greedy"` calls `predict()`, the **argmax**, which is deterministic. `mode="sample"` calls `sample()`, which uses `random.choices` with the CPT row as weights, so a word is chosen **in proportion to its probability**. Greedy always picks the same token for a given context; sampling draws from the whole distribution, so low-probability continuations still appear |
| **7. Unseen context?** | `distribution()` returns `{}`, and `predict()`/`sample()` return `None`. `generate()` then stops with reason `"unseen"` instead of crashing or inventing a word. For example, `P(. \| bird) = {}`. In second order this also happens for known words in an unseen pair (`sat the`). Smoothing (add-one or back-off) would be the fix if non-zero probabilities were wanted |

## Part VII and Question 8: Normalisation test

Every row sums to 1.000000 for both orders (`results.txt`), and `test_rows_sum_to_one` asserts this to
12 decimal places.

**Q8. A total of 0.87** means the row is not a probability distribution: 13% of the mass is missing.
Likely causes: the wrong denominator (for example dividing by all occurrences of `w`, including ones
with no successor counted), next-tokens such as `<END>` being dropped from the counts, an off-by-one
loop, or smoothing that reserves mass for unseen words without assigning it. Sampling would still "work",
because `random.choices` renormalises weights, so the bug would be **silent** without this test.

## Part VIII and Question 9: Next-word prediction

| Context | P(next \| context) | argmax |
|---|---|---|
| `<START>` | the 1.0 | the |
| `the` | cat .25, dog .25, mat .167, rug .167, park .167 | cat (tie with dog) |
| `cat` | sat .667, ran .333 | sat |
| `dog` | sat .667, ran .333 | sat |
| `sat` | on 1.0 | on |
| `ran` | to 1.0 | to |
| `on` / `to` | the 1.0 | the |
| `bird` | (no row) | None |

**Q9.** Mostly, but not always. After "sat on the" I expect *mat* or *rug*. The first-order model only
sees `the`, so its top prediction is *cat* (tied with *dog*), and it gives *park* 0.167 even though
"sat on the park" is odd. The model reports **corpus frequencies under its independence assumption**,
not meaning. Human expectations use the whole context, world knowledge and grammar. A probability
model is only as good as its data and its conditioning set.

## Part IX: Generated text

Twenty samples from each model are in `generated_sentences.txt`. Examples (first order):

```
the cat sat on the rug          (training sentence)
the cat ran to the mat          (novel, grammatical)
the cat sat on the park         (novel, odd)
the rug                         (novel, fragment)
the cat sat on the dog sat on the dog ran to the cat ran to the rug   (run-on)
```

## Part X and Question 10: Greedy vs sampling

| Mode | Order 1 | Order 2 |
|---|---|---|
| Greedy (×5) | `the cat sat on the cat sat on …` ×5, hits `max_len` and **never ends** | `the cat sat on the mat` ×5 |
| Sampling (×5) | 5 different outputs, from 4 to 14 words | 4 distinct training sentences |

**Q10. Sampling produces far more variation.** Greedy decoding is a deterministic function of the
context, so it produces the same sentence every time, and it ignores everything except the top token.
In first order it even loops forever, because the argmax path `the → cat → sat → on → the` is a cycle
that never reaches `<END>`. Sampling explores the whole distribution, so lower-probability branches
(`dog`, `ran`, `park`) appear in proportion to their probabilities.

## Question 11: First vs second order

| Aspect | First order | Second order |
|---|---|---|
| Graph | Chain Xₜ₋₁ → Xₜ: one parent per node | Xₜ₋₂ → Xₜ ← Xₜ₋₁: two parents per node |
| CPT | One row per word: \|V\| rows × \|V\| columns (121 cells) | One row per word *pair*: \|V\|² rows × \|V\| (1331 cells) |
| Context | 1 previous token | 2 previous tokens. For example, it distinguishes `on the` → {mat, rug} from `to the` → {park} |
| Data needed | Each pair must be observed | Each triple must be observed: exponentially more data for the same coverage |

## Part XIII: Comparison (1000 samples each, seed 407)

| Metric | Order 1 | Order 2 |
|---|---|---|
| Observed contexts (CPT rows) | 11 | 15 |
| Possible contexts | 11 | 121 |
| **Zero-probability (unseen) contexts** | 0 | **106** |
| Full CPT size (cells) / free parameters | 121 / 110 | 1331 / 1210 |
| **Non-zero estimated parameters** | 17 | 19 |
| **Distinct sentences** | **188** | **6** |
| Novel sentences (not in training) | 182 | 0 |
| Samples that are training sentences | 13.7% | 100% |
| Max length (tokens) | 46 | 6 |

**Coherence.** All second-order samples are grammatical, because they are exactly the 6 training
sentences. First-order samples include sensible novel sentences ("the cat ran to the mat"), semantic
oddities ("the cat sat on the park"), fragments ("the rug") and run-ons. Order 1 is diverse but
incoherent; order 2 is coherent but only memorises. With 6 sentences, every observed pair-context has
essentially one continuation path.

## Question 12: Context vs estimability

More context helps because the true distribution P(Xₜ | x₁..ₜ₋₁) depends on more than one word. Conditioning on `on the`
instead of `the` removes impossible continuations (`cat`, `park`) and sharpens the prediction. But the
CPT grows as |V|^k × |V| (121 → 1331 cells here, and 10¹² for k = 4 with a 1000-word vocabulary), while the
data stays fixed. Most rows are never observed (106 of 121 contexts here), and the observed ones are
estimated from one or two counts. The model then either assigns zero probability to anything new or
just reproduces the training data (0 novel sentences). This is the bias–variance trade-off: higher
order means lower bias and much higher variance, or sparsity.

## Part XIV: Connection to modern LMs

A modern LM keeps the same objective, ∏ₜ P(xₜ | x₁..ₜ₋₁), with the first factor read as P(x₁ | `<START>`).
It replaces the explicit CPT with a neural network that shares parameters across contexts. That is how it
escapes the |V|^k blow-up seen in Q12: similar contexts get similar predictions without each row
being counted separately. Generation is the same procedure: sample, append, repeat.

## Question 13: Why Approach B is preferable

- **Specifying behaviour.** "Estimate P(Xₜ | Xₜ₋₁) from counts, sample until `<END>`" defines what
  correct means. "Write a language model" lets the LLM choose the model, a library or a pretrained
  network, and leaves nothing to check against.
- **Understanding the representation.** Knowing that rows are contexts and columns are next tokens let
  me answer Q4–Q7 by pointing at code, and predict that greedy decoding would loop.
- **Validating the implementation.** The generated code ran, but one generated test was wrong (see
  `PROMPTS.md`). Running and inspecting caught that; trusting it would not have.
- **Testing invariants.** Row sums equal 1, counts equal an independent oracle, sampling frequencies
  match the CPT, and triples marginalise to pairs. These follow from the model, not from the code, so
  they catch silent bugs (Q8).
- **Implementation vs model.** The model is the factorisation and its CPTs. Dicts, `random.choices`
  and tie-breaking are implementation choices. Keeping them separate made the second-order change a
  context-length change only.

## Question 14: What the Bayesian network added

1. **A representation of dependencies.** The arrows say exactly what each word conditions on, and
   the graph *is* the modelling assumption.
2. **A factorisation of the joint.** P(X₁..X_T) = ∏ P(Xₜ | parents), so the probability of any
   sentence is computable (`sentence_prob`: "the cat sat on the mat" = 1/36).
3. **Interpretable conditional probabilities.** Each CPT row is a readable next-word distribution.
4. **A principled generation method.** Ancestral sampling in topological order is exactly
   autoregressive generation.
5. **Independence assumptions and the effect of context.** Adding a parent (order 2) visibly changes
   the graph, the CPT size and the data requirements (Q11–Q12).
6. **A testable specification.** Normalisation and marginalisation tests come directly from the BN
   semantics.

## Reflection on LLM use

The LLM wrote the class structure, the sampling/greedy logic, the second-order generalisation and
most of the test scaffolding. I wrote the specification, the hand CPT and the expected changes for
second order before prompting. I validated the output by running invariant tests against independent
oracles. One concrete correction: a generated test assumed sampled sentences always end within 20
tokens. It failed, and analysis showed that the model was right and the test was wrong (details in
`PROMPTS.md`).
