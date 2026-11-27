"""Parts VII-XIII: normalisation, prediction, generation, greedy vs sampling, order comparison.

Run: python experiments.py   (writes results.txt and generated_sentences.txt)
"""
import random
import sys
from collections import Counter

from ngram_lm import CORPUS, END, START, AutoregressiveLM

SEED = 407


def normalisation(lm):
    print(f"-- Part VII: sum_v P(v | w) for every context (order {lm.order})")
    for ctx, row in sorted(lm.probabilities.items()):
        print(f"   {' '.join(ctx):<16} {sum(row.values()):.6f}")


def predictions(lm, contexts):
    print(f"-- Part VIII: next-word distributions and argmax (order {lm.order})")
    for c in contexts:
        dist = lm.distribution(c)
        shown = ", ".join(f"{w}={p:.3f}" for w, p in sorted(dist.items(), key=lambda x: -x[1]))
        print(f"   P(. | {c:<12}) = {{{shown}}}  ->  argmax = {lm.predict(c)}")


def gen(lm, mode, n, rng, max_len=20):
    out = []
    for _ in range(n):
        toks, why = lm.generate(mode, rng, max_len)
        out.append(" ".join(toks) + ("" if why == "end" else f"  [{why}]"))
    return out


def stats(lm, rng, n=1000):
    """Numbers for Part XIII."""
    ctx_vocab = [START] + [w for w in lm.vocab() if w != START]  # tokens that can be context
    next_vocab = lm.vocab() + [END]
    possible_ctx = len(ctx_vocab) ** lm.order
    full_cpt = possible_ctx * len(next_vocab)
    observed_ctx = len(lm.counts)
    samples = [" ".join(lm.generate("sample", rng, 50)[0]) for _ in range(n)]
    train = set(CORPUS)
    bigrams = Counter(b for s in samples for b in zip(s.split(), s.split()[1:]))
    return {
        "observed contexts (CPT rows)": observed_ctx,
        "possible contexts |V_ctx|^k": possible_ctx,
        "zero-probability (unseen) contexts": possible_ctx - observed_ctx,
        "full CPT size (cells)": full_cpt,
        "free parameters of full CPT (cells - rows)": full_cpt - possible_ctx,
        "non-zero estimated parameters": lm.num_parameters(),
        f"distinct sentences in {n} samples": len(set(samples)),
        f"novel (not in training) distinct sentences": len(set(samples) - train),
        f"share of samples that are training sentences": f"{sum(s in train for s in samples)/n:.3f}",
        "mean sentence length (tokens)": f"{sum(len(s.split()) for s in samples)/n:.2f}",
        "max sentence length (tokens)": max(len(s.split()) for s in samples),
        "distinct bigrams in samples": len(bigrams),
    }


def main(out=sys.stdout):
    sys.stdout = out
    rng = random.Random(SEED)
    lm1, lm2 = AutoregressiveLM(CORPUS, 1), AutoregressiveLM(CORPUS, 2)

    normalisation(lm1)
    normalisation(lm2)
    print()
    predictions(lm1, ["<START>", "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "bird"])
    predictions(lm2, ["<START> the", "the cat", "the dog", "on the", "to the", "dog ran"])
    print()

    print("-- Part X: greedy vs sampling (5 each)")
    for lm in (lm1, lm2):
        print(f"   order {lm.order}, greedy:")
        for s in gen(lm, "greedy", 5, rng):
            print(f"      {s}")
        print(f"   order {lm.order}, sampling:")
        for s in gen(lm, "sample", 5, rng):
            print(f"      {s}")
    print()

    print("-- Part XIII: first- vs second-order comparison")
    s1, s2 = stats(lm1, random.Random(SEED)), stats(lm2, random.Random(SEED))
    print(f"   {'metric':<48}{'order 1':>10}{'order 2':>10}")
    for k in s1:
        print(f"   {k:<48}{s1[k]!s:>10}{s2[k]!s:>10}")
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    with open("results.txt", "w") as f:
        main(f)
    # Part IX: 20 sampled sentences per model
    rng = random.Random(SEED)
    with open("generated_sentences.txt", "w") as f:
        for order in (1, 2):
            f.write(f"# order {order}: 20 sampled sentences (seed {SEED})\n")
            for i, s in enumerate(gen(AutoregressiveLM(CORPUS, order), "sample", 20, rng), 1):
                f.write(f"{i:2d}. {s}\n")
            f.write("\n")
    print(open("results.txt").read())
    print(open("generated_sentences.txt").read())
