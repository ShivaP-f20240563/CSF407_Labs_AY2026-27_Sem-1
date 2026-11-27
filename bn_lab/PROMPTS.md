# LLM Prompt Log (BN Lab)

LLM used: Claude. The model was specified in `DESIGN.md` before any code was requested.

## Prompt 1: first-order model (Part V)

> Write a simple Python implementation of a first-order autoregressive language model. The model
> should: 1. take a list of tokenised sentences as training data; 2. count transitions between
> consecutive tokens; 3. construct the conditional distribution P(Xt | Xt−1); 4. display the
> probabilities for a specified previous token; 5. predict the most probable next token; 6. generate a
> sentence by repeatedly sampling the next token; 7. stop when the <END> token is generated.
> Do not use a machine-learning library or a pretrained language model. Use ordinary Python data
> structures and random sampling.

Added constraints: lower-case input, add `<START>`/`<END>`, store contexts as tuples, return an empty
distribution for an unseen context instead of raising, and support a `greedy` mode (Part X).

## Prompt 2: tests (Part VII)

> Write unittest tests that check probabilistic invariants: every CPT row sums to 1, the counts equal
> an independent bigram count, the CPT equals my hand-computed table, sampling frequencies match the
> CPT, and generated sentences only use observed transitions.

## Prompt 3: second order (Part XII)

Before prompting, I wrote down what *should* change: the context becomes the pair (Xₜ₋₂, Xₜ₋₁), the
counts become triple counts, there are two `<START>` pads, and the CPT has |V|² rows instead of |V|. The
sampling and generation logic should be unchanged.

> Modify the existing first-order autoregressive model into a second-order model. The model should
> estimate P(Xt | Xt−2, Xt−1). Represent the model using counts of observed triples and use these
> counts to construct conditional probability distributions. Do not replace the model with a neural
> network or a pretrained language model.

The generated change generalised the class to an `order` parameter. I checked that the diff only
touched the context length and the padding, and added second-order tests: triple counts against an
independent oracle, marginalising triples recovers the bigram counts, and only the last two tokens
matter.

## Code I inspected or corrected

1. **Test assumption corrected.** The generated test
   `test_sampled_sentences_end_and_use_only_observed_bigrams` asserted that every sampled sentence
   ends with `<END>` within the default `max_len=20`. It **failed**. Inspection showed the model is
   right and the test was wrong: the first-order chain returns from `the` to `the` with probability 0.5
   (via `cat/dog → sat → on` or `→ ran → to`), so sentences of 20+ tokens are legitimate samples (the
   longest in 1000 samples was 46 tokens). I raised the cap to 200 in that test (P(hit) ≈ 0.5⁵⁰) and
   documented why.
2. **Generation needed a stopping guard.** The specification said "stop when `<END>` is generated",
   but greedy first-order decoding never generates `<END>`: `the → cat → sat → on → the → …`. I kept
   the `max_len` guard and made `generate()` return a stop reason (`end`, `max_len` or `unseen`), so
   the loop is visible in the results instead of hidden.
3. **Tie-breaking made explicit.** `predict("the")` is a tie between `cat` and `dog` (0.25 each). The
   result depended on dictionary insertion order, so I made it alphabetical, which keeps the tests
   reproducible.
4. **Hand CPT vs handout.** The handout's example gives P(cat | the) = 3/5, but on this corpus it is
   3/12, because `the` also precedes mat, rug and park. The code matches my hand table (`DESIGN.md`),
   and the test checks it with exact fractions.
