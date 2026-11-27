"""Autoregressive n-gram language model as a chain-structured Bayesian network.

First order:  P(x_1..x_T) = prod_t P(x_t | x_{t-1}),  x_0 = <START>
Each CPT row P(. | context) is estimated by counting:
    P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)
Standard library only (no ML library, no pretrained model).
"""
import random
from collections import Counter, defaultdict

START, END = "<START>", "<END>"

CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenize(sentence):
    return sentence.lower().split()


class AutoregressiveLM:
    def __init__(self, sentences):
        self.order = 1
        # counts[context][next] = C(context, next); context is a tuple of the previous tokens
        self.counts = defaultdict(Counter)
        for s in sentences:
            toks = [START] * self.order + tokenize(s) + [END]
            for i in range(self.order, len(toks)):
                self.counts[tuple(toks[i - self.order:i])][toks[i]] += 1
        # probabilities[context][next] = P(next | context): normalised CPT rows
        self.probabilities = {
            ctx: {w: c / sum(nxt.values()) for w, c in nxt.items()}
            for ctx, nxt in self.counts.items()
        }

    def _ctx(self, context):
        """Accept a token, a list of tokens or a tuple; keep the last `order` tokens."""
        if isinstance(context, str):
            context = context.split()
        context = [START] * self.order + list(context)
        return tuple(context[-self.order:])

    def distribution(self, context):
        """P(X_t | context). Empty dict if the context was never observed."""
        return dict(self.probabilities.get(self._ctx(context), {}))

    def show(self, context):
        dist = self.distribution(context)
        label = " ".join(self._ctx(context))
        if not dist:
            print(f"P(. | {label}): context never observed (no CPT row)")
            return
        print(f"P(. | {label}):")
        for w, p in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {w:<8} {p:.3f}")

    def predict(self, context):
        """argmax_w P(w | context); ties broken alphabetically for reproducibility."""
        dist = self.distribution(context)
        if not dist:
            return None
        return min(dist, key=lambda w: (-dist[w], w))

    def sample(self, context, rng=random):
        dist = self.distribution(context)
        if not dist:
            return None
        words = sorted(dist)
        return rng.choices(words, weights=[dist[w] for w in words])[0]

    def generate(self, mode="sample", rng=random, max_len=20):
        """Sample -> append -> sample again, until <END> or max_len tokens.

        Returns (tokens, stop_reason) with stop_reason in {"end", "max_len", "unseen"}.
        """
        history = []
        while len(history) < max_len:
            nxt = self.predict(history) if mode == "greedy" else self.sample(history, rng)
            if nxt is None:
                return history, "unseen"
            if nxt == END:
                return history, "end"
            history.append(nxt)
        return history, "max_len"

    def sentence_prob(self, sentence):
        """P(sentence) = prod of next-token conditionals, including P(<END> | last)."""
        toks, p = tokenize(sentence) + [END], 1.0
        for i, w in enumerate(toks):
            p *= self.distribution(toks[:i]).get(w, 0.0)
        return p


if __name__ == "__main__":
    lm = AutoregressiveLM(CORPUS)
    for w in ["the", "cat", "dog", "sat", "ran"]:
        lm.show(w)
        print(f"    argmax: {lm.predict(w)}")
    rng = random.Random(0)
    for _ in range(5):
        toks, why = lm.generate("sample", rng)
        print(" ".join(toks), f"[{why}]")
