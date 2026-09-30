"""
Neural Models Lab: XOR sensor-disagreement detector (CS F407).

Sections (run all:  python xor_lab.py):
    0. Linear baseline            (Task 1 prediction check)
    A. Basic learning check       (Task 4 Part A)
    B. Backpropagation check      (Task 4 Part B)
    C. Symmetry experiment        (Task 4 Part C)
    D. Activation experiment      (Task 4 Part D)
    R. Repeated-run robustness    (Task 2 validation criterion)
    E. Three-class extension      (Task 5)

Model (Task 2 design):  2 inputs -> 2 hidden (nonlinear) -> 1 logit
Loss: BCEWithLogitsLoss (sigmoid + binary cross-entropy, numerically stable)
Optimiser: full-batch Adam, lr=0.05, 3000 steps (CPU, < 1 s)
"""

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)   # float64 so finite-difference checks are meaningful
torch.set_printoptions(precision=4, sci_mode=False)

# ---------------------------------------------------------------- data
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])              # XOR / disagreement
Y3 = torch.tensor([0, 1, 1, 2])                         # 3-class extension

ACTS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
STEPS, LR = 3000, 0.05


def make_net(act="tanh", hidden=2, out=1, seed=0, const_init=None):
    """const_init=c sets every weight and bias to the constant c (symmetry test)."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, hidden), ACTS[act](), nn.Linear(hidden, out))
    if const_init is not None:
        for p in net.parameters():
            nn.init.constant_(p, const_init)
    return net


def train(net, loss_fn, target, steps=STEPS, lr=LR, opt="adam", log_grad_at=10, trace=None):
    """Full-batch training. Returns (initial_loss, final_loss, early_grad_norm)."""
    optim = (torch.optim.Adam if opt == "adam" else torch.optim.SGD)(net.parameters(), lr=lr)
    W1 = net[0].weight
    init_loss, early = None, None
    for step in range(steps):
        optim.zero_grad()
        logits = net(X)                       # forward pass
        loss = loss_fn(logits, target)        # scalar loss
        loss.backward()                       # reverse-mode AD -> .grad
        if step == 0:
            init_loss = loss.item()
        if step == log_grad_at:
            early = W1.grad.norm().item()
        if trace is not None:
            trace(step, net)
        optim.step()                          # parameter update
    with torch.no_grad():
        final = loss_fn(net(X), target).item()
    return init_loss, final, early


def binary_report(net):
    with torch.no_grad():
        p = torch.sigmoid(net(X)).squeeze(1)
    labels = (p > 0.5).long()
    correct = int((labels == Y.squeeze(1).long()).sum())
    return p, labels, correct


bce = nn.BCEWithLogitsLoss()


# ---------------------------------------------------------------- sections
def section0_linear():
    print("=" * 70, "\n0. LINEAR BASELINE (single affine map + sigmoid)")
    torch.manual_seed(0)
    lin = nn.Sequential(nn.Linear(2, 1))
    i, f, _ = train(lin, bce, Y)
    with torch.no_grad():
        p = torch.sigmoid(lin(X)).squeeze(1)
    print(f"loss {i:.4f} -> {f:.4f}  (ln 2 = 0.6931)")
    print("probabilities:", p)
    print("=> stuck near p=0.5 for all inputs: no line separates XOR.\n")


def sectionA_basic():
    print("=" * 70, "\nA. BASIC LEARNING CHECK  (2-2-1, tanh, Adam, seed 0)")
    net = make_net("tanh", seed=0)
    i, f, _ = train(net, bce, Y)
    p, labels, correct = binary_report(net)
    print(f"initial loss {i:.4f}   final loss {f:.6f}")
    print("probabilities   :", p)
    print("thresholded     :", labels)
    print(f"correct         : {correct}/4\n")
    return net


def sectionB_backprop():
    print("=" * 70, "\nB. BACKPROPAGATION CHECK")
    net = make_net("tanh", seed=0)
    net.zero_grad()
    bce(net(X), Y).backward()
    batch_grad = net[0].weight.grad.clone()
    print("dL/dW1 (mean loss over 4 examples):\n", batch_grad)

    # average of per-example gradients
    per_ex = []
    for k in range(4):
        net.zero_grad()
        bce(net(X[k:k + 1]), Y[k:k + 1]).backward()
        per_ex.append(net[0].weight.grad.clone())
    avg = torch.stack(per_ex).mean(0)
    print("mean of 4 per-example gradients:\n", avg)
    print("max |difference| =", (avg - batch_grad).abs().max().item())

    # finite-difference check on one weight
    eps = 1e-4
    with torch.no_grad():
        w = net[0].weight
        w[0, 0] += eps; lp = bce(net(X), Y).item()
        w[0, 0] -= 2 * eps; lm = bce(net(X), Y).item()
        w[0, 0] += eps
    print(f"finite difference dL/dW1[0,0] = {(lp - lm) / (2 * eps):.6f} "
          f"vs autograd {batch_grad[0, 0].item():.6f}\n")


def sectionC_symmetry():
    print("=" * 70, "\nC. SYMMETRY EXPERIMENT  (identical initial weights)")
    for act, c in (("tanh", 0.0), ("sigmoid", 0.0), ("tanh", 0.5)):
        net = make_net(act, const_init=c)
        snaps = {}

        def trace(step, n):
            if step in (0, 1, 10, 100, STEPS - 1):
                snaps[step] = n[0].weight.detach().clone()

        _, f, _ = train(net, bce, Y, trace=trace)
        _, _, correct = binary_report(net)
        print(f"-- hidden = {act}, every weight and bias initialised to {c}")
        for s, W in snaps.items():
            r0 = [round(v, 4) for v in W[0].tolist()]
            r1 = [round(v, 4) for v in W[1].tolist()]
            print(f"step {s:4d}: row0={r0}  row1={r1}  identical={torch.equal(W[0], W[1])}")
        print(f"final loss {f:.4f}, correct {correct}/4\n")


def sectionD_activations():
    print("=" * 70, "\nD. ACTIVATION EXPERIMENT  (seed 0, only activation changes)")
    rows = []
    for act in ("sigmoid", "tanh", "relu"):
        net = make_net(act, seed=0)
        _, f, g = train(net, bce, Y)
        _, _, c = binary_report(net)
        with torch.no_grad():
            pre = net[0](X)
        rows.append((act, f, c, g, pre))
    print(f"{'activation':<10} {'final loss':>12} {'4/4?':>6} {'||grad W1|| @step10':>20}")
    for act, f, c, g, _ in rows:
        print(f"{act:<10} {f:>12.6f} {('yes' if c == 4 else f'{c}/4'):>6} {g:>20.6f}")
    print("\nDiagnostic: hidden pre-activations a = W1 x + b1 after training")
    for act, _, _, _, a in rows:
        print(f"{act:<8} a =", [[round(v, 3) for v in row] for row in a.tolist()])
    print()
    return rows


def sectionR_repeats(seeds=range(20)):
    print("=" * 70, f"\nR. REPEATED RUNS  ({len(seeds)} seeds each; runs solving XOR 4/4)")
    configs = [("Adam lr=0.05", "adam", 0.05, 2), ("SGD  lr=0.5 ", "sgd", 0.5, 2),
               ("Adam lr=0.05", "adam", 0.05, 4)]
    print(f"{'optimiser':<13} {'hidden':>6} " + " ".join(f"{a:>8}" for a in ACTS))
    out = {}
    for label, opt, lr, hidden in configs:
        row = []
        for act in ACTS:
            ok = 0
            for s in seeds:
                net = make_net(act, hidden=hidden, seed=s)
                train(net, bce, Y, lr=lr, opt=opt)
                ok += binary_report(net)[2] == 4
            row.append(ok)
        out[(label, hidden)] = row
        print(f"{label:<13} {hidden:>6} " + " ".join(f"{k:>5}/{len(seeds)}" for k in row))
    print()
    return out


def sectionE_three_class():
    print("=" * 70, "\nE. THREE-CLASS EXTENSION  (2-2-3, softmax + cross-entropy)")
    net = make_net("tanh", hidden=2, out=3, seed=0)
    print("final weight matrix shape:", tuple(net[2].weight.shape), "(3 x 2)")
    ce = nn.CrossEntropyLoss()
    i, f, _ = train(net, ce, Y3)
    with torch.no_grad():
        logits = net(X)
        P = torch.softmax(logits, dim=1)
    print(f"loss {i:.4f} -> {f:.6f}")
    print("logits per example:", tuple(logits.shape))
    print("class probabilities (rows = (0,0),(0,1),(1,0),(1,1)):\n", P)
    print("predicted classes:", P.argmax(1).tolist(), " target:", Y3.tolist())
    print(f"sum of probs for (0,1): {P[1].sum().item():.8f}")
    shifted = torch.softmax(logits[1] + 100, dim=0)
    print("softmax(z + 100) for (0,1):", shifted,
          " max diff:", (shifted - P[1]).abs().max().item())

    # verify dL/dz = (p - y) / N  (mean reduction)
    z = logits.clone().requires_grad_(True)
    ce(z, Y3).backward()
    onehot = torch.nn.functional.one_hot(Y3, 3).float()
    expected = (torch.softmax(z.detach(), 1) - onehot) / 4
    print("max |dL/dz - (p - y)/N| =", (z.grad - expected).abs().max().item(), "\n")


if __name__ == "__main__":
    section0_linear()
    sectionA_basic()
    sectionB_backprop()
    sectionC_symmetry()
    sectionD_activations()
    sectionR_repeats()
    sectionE_three_class()
