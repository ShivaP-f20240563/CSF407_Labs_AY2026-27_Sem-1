"""STRIPS-style planner: logical applicability checks + breadth-first search.

A state is a frozenset of ground propositions (closed world: absent = false).
An action is applicable in S iff S |= Pre(a): every positive precondition is in S
and no negative precondition is in S. Applying it gives S' = (S - delete) | add.

Run:  python planner.py
"""
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    name: str
    pre_pos: frozenset = field(default_factory=frozenset)
    pre_neg: frozenset = field(default_factory=frozenset)
    add: frozenset = field(default_factory=frozenset)
    delete: frozenset = field(default_factory=frozenset)

    def unsatisfied(self, state):
        """Preconditions that fail in `state` (empty list = applicable)."""
        missing = [p for p in sorted(self.pre_pos) if p not in state]
        present = [f"not {p}" for p in sorted(self.pre_neg) if p in state]
        return missing + present

    def applicable(self, state):  # logical component: S |= Pre(a)
        return self.pre_pos <= state and not (self.pre_neg & state)

    def apply(self, state):  # S' = (S - Del) U Add
        return frozenset((state - self.delete) | self.add)


def act(name, pre=(), add=(), delete=(), pre_neg=()):
    return Action(name, frozenset(pre), frozenset(pre_neg), frozenset(add), frozenset(delete))


def satisfies(state, goal):
    return frozenset(goal) <= state


def bfs_plan(initial, actions, goal):
    """Return (plan, states, expanded) or (None, None, expanded) if no plan exists.

    `states` has len(plan)+1 entries: the initial state and the state after each action.
    """
    start = frozenset(initial)
    if satisfies(start, goal):
        return [], [start], 0
    parent = {start: None}  # state -> (previous state, action)
    frontier = deque([start])
    expanded = 0
    while frontier:
        s = frontier.popleft()
        expanded += 1
        for a in actions:
            if not a.applicable(s):  # logic decides what is possible
                continue
            s2 = a.apply(s)
            if s2 in parent:  # search decides what to try (skip repeats)
                continue
            parent[s2] = (s, a)
            if satisfies(s2, goal):
                return _reconstruct(parent, s2) + (expanded,)
            frontier.append(s2)
    return None, None, expanded


def _reconstruct(parent, s):
    plan, states = [], [s]
    while parent[s] is not None:
        s, a = parent[s]
        plan.append(a)
        states.append(s)
    return plan[::-1], states[::-1]


def validate(initial, plan_names, actions, goal):
    """Independent plan checker: re-executes a plan given as action names.

    Returns (ok, message, states). Reports the first unknown or inapplicable action.
    """
    table = {a.name: a for a in actions}
    s = frozenset(initial)
    states = [s]
    for i, name in enumerate(plan_names, 1):
        a = table.get(name)
        if a is None:
            return False, f"step {i}: {name} is not a permitted action", states
        bad = a.unsatisfied(s)
        if bad:
            return False, f"step {i}: {name} not applicable; unsatisfied: {', '.join(bad)}", states
        s = a.apply(s)
        states.append(s)
    if not satisfies(s, goal):
        missing = sorted(frozenset(goal) - s)
        return False, f"all steps applicable but goal not reached; missing: {', '.join(missing)}", states
    return True, f"valid plan of length {len(plan_names)}", states


def fmt(state):
    return "{" + ", ".join(sorted(state)) + "}"


def report(title, initial, actions, goal):
    print(f"== {title} ==")
    print(f"I = {fmt(frozenset(initial))}\nG = {fmt(frozenset(goal))}")
    plan, states, expanded = bfs_plan(initial, actions, goal)
    if plan is None:
        print(f"No plan found ({expanded} states expanded)\n")
        return None
    print(f"Plan ({len(plan)} actions, {expanded} states expanded):")
    print(f"  S0: {fmt(states[0])}")
    for i, (a, s) in enumerate(zip(plan, states[1:]), 1):
        print(f"  {a.name:<20} -> S{i}: {fmt(s)}")
    print()
    return plan


# ---------------------------------------------------------------- domains
LOCS = ("A", "B", "C")
LINKS = (("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"))


def warehouse_actions(links=LINKS, with_pickup=True):
    acts = [act(f"Move({x},{y})", pre=[f"At(Robot,{x})"], add=[f"At(Robot,{y})"],
                delete=[f"At(Robot,{x})"]) for x, y in links]
    for x in LOCS:
        if with_pickup:
            acts.append(act(f"PickUp(Package,{x})", pre=[f"At(Robot,{x})", f"At(Package,{x})"],
                            add=["Holding(Package)"], delete=[f"At(Package,{x})"]))
        acts.append(act(f"Drop(Package,{x})", pre=[f"At(Robot,{x})", "Holding(Package)"],
                        add=[f"At(Package,{x})"], delete=["Holding(Package)"]))
    return acts


WAREHOUSE_I = {"At(Robot,A)", "At(Package,A)"}
WAREHOUSE_G = {"At(Package,C)"}


def door_actions(with_pickkey=True):
    acts = [
        act("Move(A,B)", pre=["At(A)"], add=["At(B)"], delete=["At(A)"]),
        act("Move(B,A)", pre=["At(B)"], add=["At(A)"], delete=["At(B)"]),
        act("PickKey", pre=["At(A)", "KeyAt(A)"], add=["HasKey"], delete=["KeyAt(A)"]),
        act("Unlock", pre=["At(B)", "HasKey"], add=["Unlocked"]),
        act("Move(B,C)", pre=["At(B)", "Unlocked"], add=["At(C)"], delete=["At(B)"]),
        act("Move(C,B)", pre=["At(C)", "Unlocked"], add=["At(B)"], delete=["At(C)"]),
    ]
    return acts if with_pickkey else [a for a in acts if a.name != "PickKey"]


DOOR_I = {"At(A)", "KeyAt(A)"}
DOOR_G = {"At(C)"}


def main():
    report("Test A: warehouse", WAREHOUSE_I, warehouse_actions(), WAREHOUSE_G)
    report("Test B: warehouse without PickUp", WAREHOUSE_I, warehouse_actions(with_pickup=False),
           WAREHOUSE_G)
    report("Test C: robot-only goal At(Robot,C) (contrast with Test A)", WAREHOUSE_I,
           warehouse_actions(), {"At(Robot,C)"})
    report("Task 6(b): locked door", DOOR_I, door_actions(), DOOR_G)
    report("Task 6(c): locked door without PickKey", DOOR_I, door_actions(False), DOOR_G)

    print("== Task 6(d): validate Move(A,B), Unlock, Move(B,C) ==")
    ok, msg, _ = validate(DOOR_I, ["Move(A,B)", "Unlock", "Move(B,C)"], door_actions(), DOOR_G)
    print(f"  {'VALID' if ok else 'INVALID'}: {msg}\n")


if __name__ == "__main__":
    main()
