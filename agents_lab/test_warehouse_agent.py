"""Tests for the goal-based warehouse agent.  Run:  python -m unittest -v"""

import unittest
from collections import deque

from warehouse_agent import ACTIONS, WAREHOUSE_MAP, Warehouse, solve


def oracle_shortest(text):
    """Independent BFS distance used as a reference answer."""
    env = Warehouse(text)
    dist = {env.start: 0}
    q = deque([env.start])
    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS.values():
            n = (r + dr, c + dc)
            if env.is_free(n) and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist.get(env.goal)


class TestWarehouseAgent(unittest.TestCase):
    def assert_valid_path(self, text, traj):
        env = Warehouse(text)
        self.assertEqual(traj[0], env.start)
        self.assertEqual(traj[-1], env.goal)
        for a, b in zip(traj, traj[1:]):
            self.assertTrue(env.is_free(b), f"{b} is an obstacle")
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1, "non-unit move")

    def test_lab_map_path_is_valid_and_shortest(self):
        ok, traj, acts, _ = solve(WAREHOUSE_MAP)
        self.assertTrue(ok)
        self.assert_valid_path(WAREHOUSE_MAP, traj)
        self.assertEqual(len(acts), oracle_shortest(WAREHOUSE_MAP))

    def test_adjacent_goal(self):
        ok, _, acts, _ = solve("#####\n#SG##\n#####")
        self.assertTrue(ok)
        self.assertEqual(acts, ["Right"])

    def test_vertical_neighbour(self):
        self.assertEqual(solve("###\n#S#\n#G#\n###")[2], ["Down"])

    def test_no_path(self):
        ok, _, acts, _ = solve("#######\n#S..#G#\n#######")
        self.assertFalse(ok)
        self.assertEqual(acts, [])

    def test_multiple_routes_returns_shortest(self):
        m = ("#######\n"
             "#S...G#\n"
             "#.###.#\n"
             "#.....#\n"
             "#######")
        ok, _, acts, _ = solve(m)
        self.assertTrue(ok)
        self.assertEqual(len(acts), 4)

    def test_larger_open_map(self):
        m = "\n".join(["#" * 42] +
                      ["#S" + "." * 39 + "#"] +
                      ["#" + "." * 40 + "#"] * 18 +
                      ["#" + "." * 39 + "G#", "#" * 42])
        ok, traj, acts, _ = solve(m)
        self.assertTrue(ok)
        self.assert_valid_path(m, traj)
        self.assertEqual(len(acts), oracle_shortest(m))

    def test_invalid_map_rejected(self):
        with self.assertRaises(ValueError):
            Warehouse("#####\n#S..#\n#####")  # no goal


if __name__ == "__main__":
    unittest.main()
