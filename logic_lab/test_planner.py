"""Tests for planner.py (Task 3 tests A-C, Task 6). Run: python -m unittest -v"""
import unittest

from planner import (DOOR_G, DOOR_I, WAREHOUSE_G, WAREHOUSE_I, act, bfs_plan, door_actions,
                     satisfies, validate, warehouse_actions)


def names(plan):
    return [a.name for a in plan]


class WarehouseTests(unittest.TestCase):
    def test_A_solvable_plan_is_valid_and_shortest(self):
        plan, states, _ = bfs_plan(WAREHOUSE_I, warehouse_actions(), WAREHOUSE_G)
        self.assertEqual(names(plan), ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)",
                                       "Drop(Package,C)"])
        # Verify every step independently of the search.
        ok, msg, trace = validate(WAREHOUSE_I, names(plan), warehouse_actions(), WAREHOUSE_G)
        self.assertTrue(ok, msg)
        self.assertEqual(trace, states)

    def test_B_no_pickup_means_no_plan(self):
        plan, _, _ = bfs_plan(WAREHOUSE_I, warehouse_actions(with_pickup=False), WAREHOUSE_G)
        self.assertIsNone(plan)

    def test_C_robot_at_C_is_not_package_at_C(self):
        # Irrelevant robot-only actions: an extra Move(A,C) shortcut and a no-op Wait.
        extra = [act("Move(A,C)", pre=["At(Robot,A)"], add=["At(Robot,C)"],
                     delete=["At(Robot,A)"]),
                 act("Wait(Robot)", pre=[], add=[], delete=[])]
        acts = warehouse_actions() + extra
        plan, states, _ = bfs_plan(WAREHOUSE_I, acts, WAREHOUSE_G)
        # Goal is not satisfied merely by the robot reaching C.
        robot_only = validate(WAREHOUSE_I, ["Move(A,C)"], acts, WAREHOUSE_G)
        self.assertFalse(robot_only[0])
        self.assertIn("goal not reached", robot_only[1])
        # The final state really has the package at C, and the plan uses the shortcut correctly.
        self.assertTrue(satisfies(states[-1], WAREHOUSE_G))
        self.assertEqual(names(plan), ["PickUp(Package,A)", "Move(A,C)", "Drop(Package,C)"])
        self.assertNotIn("Wait(Robot)", names(plan))

    def test_inapplicable_hint_plan_rejected(self):
        # The handout's hint picks the package up at B, where it is not.
        ok, msg, _ = validate(WAREHOUSE_I, ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)",
                                            "Drop(Package,C)"], warehouse_actions(), WAREHOUSE_G)
        self.assertFalse(ok)
        self.assertIn("step 2", msg)
        self.assertIn("At(Package,B)", msg)

    def test_goal_already_true(self):
        plan, _, _ = bfs_plan({"At(Robot,C)", "At(Package,C)"}, warehouse_actions(), WAREHOUSE_G)
        self.assertEqual(plan, [])

    def test_negative_precondition(self):
        a = act("Enter", pre=["At(Door)"], pre_neg=["Locked"], add=["Inside"])
        self.assertTrue(a.applicable(frozenset({"At(Door)"})))
        self.assertFalse(a.applicable(frozenset({"At(Door)", "Locked"})))
        self.assertEqual(a.unsatisfied(frozenset({"At(Door)", "Locked"})), ["not Locked"])

    def test_unknown_action_rejected(self):
        ok, msg, _ = validate(WAREHOUSE_I, ["Teleport(C)"], warehouse_actions(), WAREHOUSE_G)
        self.assertFalse(ok)
        self.assertIn("not a permitted action", msg)


class LockedDoorTests(unittest.TestCase):
    def test_6b_shortest_plan(self):
        plan, states, _ = bfs_plan(DOOR_I, door_actions(), DOOR_G)
        self.assertEqual(names(plan), ["PickKey", "Move(A,B)", "Unlock", "Move(B,C)"])
        self.assertTrue(validate(DOOR_I, names(plan), door_actions(), DOOR_G)[0])
        # Exactly one At(.) holds in every reachable state on the trace.
        for s in states:
            self.assertEqual(sum(p.startswith("At(") for p in s), 1)

    def test_6c_no_pickkey_no_plan(self):
        plan, _, _ = bfs_plan(DOOR_I, door_actions(with_pickkey=False), DOOR_G)
        self.assertIsNone(plan)

    def test_6d_first_invalid_action(self):
        ok, msg, _ = validate(DOOR_I, ["Move(A,B)", "Unlock", "Move(B,C)"], door_actions(), DOOR_G)
        self.assertFalse(ok)
        self.assertEqual(msg, "step 2: Unlock not applicable; unsatisfied: HasKey")


if __name__ == "__main__":
    unittest.main()
