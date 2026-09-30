"""Task 3 tests.  Run:  python -m unittest -v"""

import unittest

from search_agent import (LAB_MAP, GridProblem, astar, bfs, euclidean, manhattan, zero)

TRIVIAL = "#####\n#SG##\n#####"

NO_SOLUTION = ("#######\n"
               "#S....#\n"
               "###.###\n"
               "#...#G#\n"
               "#######")

TWO_PATHS = ("#########\n"
             "#S......#\n"
             "#.#####.#\n"
             "#.#####.#\n"
             "#......G#\n"
             "#########")          # two routes, both of length 9

UNEQUAL_PATHS = ("#########\n"
                 "#S.....G#\n"      # short route: 6 moves
                 "#.#####.#\n"
                 "#.......#\n"      # long route: 10 moves
                 "#########")


class SearchTests(unittest.TestCase):
    def check_path(self, text, res):
        p = GridProblem(text)
        self.assertEqual(res.path[0], p.initial)
        self.assertEqual(res.path[-1], p.goal)
        for a, b in zip(res.path, res.path[1:]):
            self.assertTrue(p._free(b))
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
        self.assertEqual(len(res.path) - 1, res.length)

    # Test 1: original warehouse
    def test_original_warehouse(self):
        p = GridProblem(LAB_MAP)
        a, b = astar(p), bfs(p)
        self.assertTrue(a.found)
        self.check_path(LAB_MAP, a)
        self.assertEqual(a.length, b.length)      # A* with admissible h is optimal
        self.assertLessEqual(a.expanded, b.expanded)

    # Test 2: trivial case
    def test_trivial(self):
        res = astar(GridProblem(TRIVIAL))
        self.assertTrue(res.found)
        self.assertEqual(res.actions, ["Right"])

    # Test 3: no solution -> terminates with failure
    def test_no_solution(self):
        for algo in (astar, bfs):
            res = algo(GridProblem(NO_SOLUTION))
            self.assertFalse(res.found)
            self.assertIsNone(res.length)

    # Test 4: alternative paths -> shortest returned
    def test_alternative_paths_equal(self):
        res = astar(GridProblem(TWO_PATHS))
        self.check_path(TWO_PATHS, res)
        self.assertEqual(res.length, 9)

    def test_alternative_paths_unequal(self):
        res = astar(GridProblem(UNEQUAL_PATHS))
        self.check_path(UNEQUAL_PATHS, res)
        self.assertEqual(res.length, 6)

    # Admissible heuristics must all give optimal length
    def test_admissible_heuristics_optimal(self):
        p = GridProblem(LAB_MAP)
        opt = bfs(p).length
        for h in (manhattan, zero, euclidean):
            self.assertEqual(astar(p, h).length, opt, h.__name__)

    def test_invalid_map(self):
        with self.assertRaises(ValueError):
            GridProblem("####\n#S.#\n####")


if __name__ == "__main__":
    unittest.main()
