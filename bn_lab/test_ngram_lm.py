"""Probabilistic-invariant tests for ngram_lm.py. Run: python -m unittest -v"""
import random
import unittest
from collections import Counter
from fractions import Fraction

from ngram_lm import CORPUS, END, START, AutoregressiveLM

# Hand-computed CPT rows from DESIGN.md (Question 3), as exact fractions.
HAND_CPT = {
    "the": {"cat": Fraction(3, 12), "dog": Fraction(3, 12), "mat": Fraction(2, 12),
            "rug": Fraction(2, 12), "park": Fraction(2, 12)},
    "cat": {"sat": Fraction(2, 3), "ran": Fraction(1, 3)},
    "dog": {"sat": Fraction(2, 3), "ran": Fraction(1, 3)},
    "sat": {"on": 1}, "ran": {"to": 1}, START: {"the": 1},
}


class FirstOrderTests(unittest.TestCase):
    def setUp(self):
        self.lm = AutoregressiveLM(CORPUS)

    def test_rows_sum_to_one(self):
        for ctx, row in self.lm.probabilities.items():
            self.assertAlmostEqual(sum(row.values()), 1.0, places=12, msg=ctx)

    def test_probabilities_in_unit_interval(self):
        for row in self.lm.probabilities.values():
            self.assertTrue(all(0 < p <= 1 for p in row.values()))

    def test_matches_hand_cpt(self):
        for ctx, row in HAND_CPT.items():
            got = self.lm.distribution(ctx)
            self.assertEqual(set(got), set(row), ctx)
            for w, p in row.items():
                self.assertAlmostEqual(got[w], float(p), places=12)

    def test_counts_match_independent_bigram_count(self):
        oracle = Counter()
        for s in CORPUS:
            t = [START] + s.split() + [END]
            oracle.update(zip(t, t[1:]))
        mine = Counter({(c[0], w): n for c, row in self.lm.counts.items() for w, n in row.items()})
        self.assertEqual(mine, oracle)

    def test_end_is_never_a_context(self):
        self.assertNotIn((END,), self.lm.counts)

    def test_predict_is_argmax(self):
        self.assertEqual(self.lm.predict("cat"), "sat")
        self.assertEqual(self.lm.predict("the"), "cat")  # tie cat/dog -> alphabetical

    def test_unseen_context(self):
        self.assertEqual(self.lm.distribution("bird"), {})
        self.assertIsNone(self.lm.predict("bird"))
        self.assertIsNone(self.lm.sample("bird"))

    def test_sentence_prob(self):
        # 1 * 3/12 * 2/3 * 1 * 1 * 2/12 * 1
        self.assertAlmostEqual(self.lm.sentence_prob("the cat sat on the mat"), 1 / 36)
        self.assertEqual(self.lm.sentence_prob("the cat sat the mat"), 0.0)

    def test_sampling_frequencies_match_cpt(self):
        rng, n = random.Random(1), 60000
        freq = Counter(self.lm.sample("the", rng) for _ in range(n))
        for w, p in self.lm.distribution("the").items():
            self.assertAlmostEqual(freq[w] / n, p, delta=0.01)

    def test_sampled_sentences_end_and_use_only_observed_bigrams(self):
        rng = random.Random(2)
        for _ in range(200):
            # Each visit to "the" returns to "the" with prob 0.5, so long outputs are legitimate;
            # a large cap makes hitting it negligible (0.5^50).
            toks, why = self.lm.generate("sample", rng, max_len=200)
            self.assertEqual(why, "end")
            t = [START] + toks + [END]
            for a, b in zip(t, t[1:]):
                self.assertGreater(self.lm.counts[(a,)][b], 0)

    def test_greedy_first_order_loops(self):
        toks, why = self.lm.generate("greedy", max_len=20)
        self.assertEqual(why, "max_len")
        self.assertEqual(toks[:5], ["the", "cat", "sat", "on", "the"])


class SecondOrderTests(unittest.TestCase):
    def setUp(self):
        self.lm = AutoregressiveLM(CORPUS, order=2)

    def test_rows_sum_to_one(self):
        for ctx, row in self.lm.probabilities.items():
            self.assertEqual(len(ctx), 2)
            self.assertAlmostEqual(sum(row.values()), 1.0, places=12, msg=ctx)

    def test_counts_match_independent_triple_count(self):
        oracle = Counter()
        for s in CORPUS:
            t = [START, START] + s.split() + [END]
            oracle.update(zip(t, t[1:], t[2:]))
        mine = Counter({c + (w,): n for c, row in self.lm.counts.items() for w, n in row.items()})
        self.assertEqual(mine, oracle)

    def test_context_disambiguates_the(self):
        self.assertEqual(self.lm.distribution("on the"), {"mat": 0.5, "rug": 0.5})
        self.assertEqual(self.lm.distribution("to the"), {"park": 1.0})
        self.assertEqual(self.lm.distribution([]), {"the": 1.0})  # (<START>, <START>)

    def test_only_last_two_tokens_matter(self):
        self.assertEqual(self.lm.distribution("the dog sat on the"),
                         self.lm.distribution("on the"))

    def test_unseen_pair_context(self):
        self.assertEqual(self.lm.distribution("sat the"), {})  # both words known, pair unseen

    def test_greedy_terminates(self):
        self.assertEqual(self.lm.generate("greedy"), (["the", "cat", "sat", "on", "the", "mat"],
                                                      "end"))

    def test_marginalising_matches_first_order(self):
        # sum_a C(a, b, c) over a = C(b, c): triples collapse to the first-order pair counts.
        first = AutoregressiveLM(CORPUS, order=1)
        pairs = Counter()
        for (a, b), row in self.lm.counts.items():
            for c, n in row.items():
                pairs[(b, c)] += n
        for (b,), row in first.counts.items():
            for c, n in row.items():
                self.assertEqual(pairs[(b, c)], n)

    def test_bad_order(self):
        with self.assertRaises(ValueError):
            AutoregressiveLM(CORPUS, order=0)


if __name__ == "__main__":
    unittest.main()
