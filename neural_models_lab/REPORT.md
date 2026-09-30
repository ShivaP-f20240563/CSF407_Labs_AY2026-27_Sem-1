# Neural Models Lab Report: XOR, Backprop, Symmetry, Activations, Softmax

**Course:** CS F407 Artificial Intelligence

| File | Contents |
|---|---|
| `DESIGN.md` | Task 1 (problem specification, linear-separability argument) and Task 2 (model design, validation criteria) |
| `xor_lab.py` | All experiments: linear baseline, Parts A–D, repeated runs, three-class extension |
| `results.txt` | Full output of `python xor_lab.py` (float64, seeds fixed, CPU) |
| `PROMPTS.md` | Exact LLM prompts and the corrections made |

```bash
pip install torch numpy
python xor_lab.py          # about 3 minutes on CPU (most of it is the repeated-runs section)
```

## Task 3: Generated Code, Inspected

| Step | Where it happens in `train()` |
|---|---|
| Forward pass | `logits = net(X)` |
| Scalar loss | `loss = loss_fn(logits, target)`, with `BCEWithLogitsLoss` (mean over 4 examples) |
| Reverse-mode AD | `loss.backward()` fills `p.grad` for every parameter |
| Optimiser update | `optim.step()` (Adam) |

**Think about it: what can be verified without running the code, and what needs execution?**

- *From reading:* the architecture (2-2-1), the activation, the loss/output pairing, the data and
  labels, that `zero_grad` → `backward` → `step` are in the right order, and that the seed is set.
- *Only by running:* whether the loss actually falls, whether all 4 predictions are right, the size
  of the gradients, sensitivity to the seed, and numerical issues (see the float32 finite-difference
  discrepancy in `PROMPTS.md`).

## Task 4: Results

### Linear baseline (Task 1 prediction)

The loss goes from 0.7309 to **0.6931 = ln 2**, and all four probabilities are **0.5000**. The
prediction holds: an affine map plus a sigmoid cannot represent XOR.

### Part A: Basic learning check (2-2-1, tanh, Adam lr 0.05, 3000 steps, seed 0)

| | |
|---|---|
| Initial loss | 0.7102 |
| Final loss | 0.000108 |
| Probabilities | (0,0): 0.0001, (0,1): 0.9998, (1,0): 0.9999, (1,1): 0.0001 |
| Thresholded | 0, 1, 1, 0 → **4/4 correct** |

No engineering settings had to change for this seed. However, see the repeated-runs results below.

### Part B: Backpropagation check

`net[0].weight.grad` holds **∂L/∂W⁽¹⁾**: entry (i, j) says how much the mean loss changes per unit
change in the weight from input j to hidden unit i, at the current parameters. At initialisation (seed 0):

```
dL/dW1 = [[0.0182, 0.0358],
          [0.0170, 0.0028]]
```

- **Mean of the 4 per-example gradients:** identical, with a maximum difference of 0.0. The loss is a
  *mean* L = ¼ Σₖ Lₖ, and differentiation is linear, so ∂L/∂W = ¼ Σₖ ∂Lₖ/∂W. The batch gradient is
  the average of the example-wise gradients.
- **Central finite difference** for W⁽¹⁾[0,0]: 0.018180, and autograd gives 0.018180. They agree.

### Part C: Symmetry experiment

| Initialisation | Row 0 of W⁽¹⁾ over training | Row 1 | Identical? | Final loss | Correct |
|---|---|---|---|---|---|
| All weights and biases = 0, tanh | stays [0, 0] | stays [0, 0] | **Yes, at every step** | 0.6931 | 2/4 |
| All = 0, sigmoid | stays [0, 0] | stays [0, 0] | **Yes** | 0.6931 | 2/4 |
| All = 0.5, tanh (extra) | 0.5 → 0.45 → 0.02 → −3.94 → −7.77 | same values | **Yes** | 0.4774 | 3/4 |

**Explanation:** if both hidden units start with identical incoming and outgoing weights, they
compute the same activation on every input. Backprop then gives them the same error signal
(δ₁ = δ₂), so they receive **identical gradients** and identical updates. The symmetry holds for
all time. In effect the network has only one hidden unit, and one unit cannot solve XOR.

With *exact zeros* it is even worse. W⁽²⁾ = 0 means no error signal reaches W⁽¹⁾ at all, and for
tanh h = tanh(0) = 0 means W⁽²⁾ gets no gradient either. The network sits at a stationary point at
p = 0.5. The constant-0.5 run shows the general case: the rows *do* move, but always together.
Random initialisation breaks the symmetry.

### Part D: Activation experiment (seed 0, only the activation changes)

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W⁽¹⁾L‖₂ (step 10) |
|---|---|---|---|
| Sigmoid | 0.000284 | Yes | 0.000703 |
| Tanh | 0.000108 | Yes | 0.013164 |
| ReLU | 0.693147 | No (2/4) | 0.004450 |

Hidden pre-activations **a = W⁽¹⁾x + b⁽¹⁾** after training:

```
sigmoid  [[-5.2, 5.6], [4.8, 16.9], [-15.3, -5.2], [-5.3, 6.1]]   large |a|: units operate in saturation
tanh     [[2.5, 2.5], [-2.6, 7.9], [7.6, -2.6], [2.6, 2.8]]
relu     [[0.13, 0.28], [0.12, 0.39], [0.33, 0.05], [0.32, 0.16]]  all > 0 on all 4 inputs
```

**Interpretation for this experiment only (not a universal ranking):**

- **Tanh** had the largest early gradient, about 19× the sigmoid one. tanh′(0) = 1 while σ′(0) = 0.25,
  and tanh is zero-centred, so more error signal reaches W⁽¹⁾. It reached the lowest loss.
- **Sigmoid** started with a small gradient (at most 0.25 per layer) but still solved XOR with this
  seed. By the end its pre-activations are large, so its units are saturated with σ′ ≈ 0. That is
  what a confident, converged solution looks like.
- **ReLU** failed with this seed, but *not* because of dead units. All its pre-activations are
  **positive on all four inputs**, so both ReLUs act as the identity on the data. The network is
  effectively *linear* again and gets stuck at ln 2, exactly like the linear baseline.

**Think about it: distinguishing the two small-gradient mechanisms.** Look at the pre-activations `a`:

- **Saturated sigmoid:** |a| is large (for example > 5) and the activation is close to 0 or 1, so
  σ′(a) = σ(1−σ) ≈ 0.
- **Dead ReLU:** a < 0 on every input, the activation is exactly 0 and the derivative is exactly 0.

The first shrinks the gradient smoothly and could recover. The second cuts it off completely for
those inputs. Logging `a` per unit and per input (as `xor_lab.py` does) tells them apart.

### Repeated runs (validation criterion 4)

Runs out of 20 random seeds that reach 4/4 correct:

| Optimiser | Hidden units | Sigmoid | Tanh | ReLU |
|---|---|---|---|---|
| Adam lr 0.05 | 2 | 7/20 | 10/20 | 6/20 |
| SGD lr 0.5 | 2 | 16/20 | 14/20 | 5/20 |
| Adam lr 0.05 | 4 (extra) | 19/20 | 18/20 | 10/20 |

A 2-2-1 network *can* represent XOR, but its loss surface has local minima and plateaus, so
**a single successful run is not evidence of reliable learning**. Plain SGD with a larger step
escaped these traps more often than Adam at lr 0.05. Widening the hidden layer to 4 units (extra
capacity, which smooths the loss landscape) raised success to 18–19/20 for sigmoid and tanh. This is an engineering
observation (it depends on the optimiser and learning rate). The scientific claim (a hidden
nonlinearity is necessary and sufficient to represent XOR) is unchanged.

## Task 5: Three-Class Extension (2-2-3, softmax + cross-entropy)

**Predictions before running:**

1. **Final weight matrix shape:** 3 × 2 (3 logits × 2 hidden units), plus a bias of size 3. *Confirmed:* `(3, 2)`.
2. **Logits per example:** 3, so the output is 4 × 3. *Confirmed.*
3. **Softmax sums to 1:** pₖ = e^{zₖ} / Σⱼ e^{zⱼ}. All terms share the same denominator, which is their sum.
4. **Logit gradient p − y:** L = −log p_c = −z_c + log Σⱼ e^{zⱼ}, so ∂L/∂zₖ = pₖ − 1[k = c] = pₖ − yₖ.

**Results** (loss 1.0240 → 0.000030):

| Input | P(class 0) | P(class 1) | P(class 2) | Predicted | Target |
|---|---|---|---|---|---|
| (0,0) | 1.0000 | 0.0000 | 0.0000 | 0 | 0 |
| (0,1) | 0.0000 | 1.0000 | 0.0000 | 1 | 1 |
| (1,0) | 0.0000 | 1.0000 | 0.0000 | 1 | 1 |
| (1,1) | 0.0000 | 0.0000 | 1.0000 | 2 | 2 |

- The probabilities for (0,1) sum to **1.00000000**.
- `softmax(z + 100)` is unchanged, with a maximum difference of about 8e-20. Adding c multiplies the
  numerator and denominator by e^c, which cancels.
- The autograd dL/dz equals (p − y)/N to within about 3e-21. The /N comes from the mean reduction.

**Why stable implementations subtract max(z):** mathematically it changes nothing (shift
invariance). Numerically it keeps every exponent ≤ 0, so e^{z} never overflows (e.g. e^{1000} = inf
gives inf/inf = NaN), and at least one term equals 1, so the denominator never underflows to 0.

**Think about it: scaling to next-token prediction.** What stays mathematically identical: softmax
over logits, cross-entropy = −log p(correct token), the gradient p − y, the one-hot target, shift
invariance and max-subtraction. What changes dramatically: K goes from 3 to about 50,000–200,000,
so the output matrix and softmax dominate compute and memory. The hidden layer becomes a deep
transformer producing context-dependent representations. Training uses mini-batches of sequences,
and tricks such as mixed precision, fused softmax-CE kernels, and tied embeddings become necessary.

## Reflection Questions

**1. Depth vs. nonlinearity.** Depth alone does nothing: stacked affine layers collapse into one affine
map, and the linear model stays at ln 2. The *nonlinearity* between layers is what lets the hidden
layer build new features (roughly OR and AND) in which XOR becomes linearly separable. The ReLU run
that stayed in its linear region demonstrated this by accident. When the nonlinearity was not used,
the network behaved exactly like the linear model.

**2. Evidence of a useful learning signal, not just a nonzero gradient.** The loss fell from 0.71 to
1e-4. All four outputs moved to the correct side of 0.5 with high confidence. The autograd gradient
matched a finite-difference estimate. The two hidden rows became different, so the units learned
distinct features. By contrast, the ReLU seed-0 run had a nonzero early gradient (0.0045) but never
left ln 2. A nonzero gradient is necessary, not sufficient.

**3. Why identical or zero initialisation prevents distinct features.** Identical units compute
identical outputs, receive identical backpropagated errors, and so get identical updates. By
induction they stay identical forever: rows 0 and 1 were equal at every logged step. With all
zeros, the gradient into W⁽¹⁾ is also exactly zero because W⁽²⁾ = 0.

**4. Activation vs. gradient.** *Engineering observation:* at step 10, ‖∇W⁽¹⁾‖ was 0.0132 for tanh,
0.0045 for ReLU and 0.0007 for sigmoid. *Scientific explanation:* the gradient into W⁽¹⁾ is scaled by
f′(a). At initialisation tanh′ ≈ 1, σ′ ≤ 0.25, and ReLU′ ∈ {0, 1}. Over training, sigmoid and tanh
units saturate (f′ → 0) as they become confident. ReLU units never saturate on the positive side but
give exactly zero gradient when a < 0.

**5. Why output layer and loss are chosen together.** The output layer defines a probability model and
the loss should be its negative log-likelihood. Sigmoid + BCE (Bernoulli) and softmax + CE
(categorical) both give the clean gradient p − y, which cancels the saturation of the output
nonlinearity. Mismatches, such as sigmoid + MSE, reintroduce σ′ into the gradient and can stall
learning when the model is confidently wrong. Softmax with BCE would treat mutually exclusive classes
as independent.

**6. LLM productivity vs. human verification.** *Productivity:* the LLM produced working PyTorch
boilerplate (the training loop, per-example gradient loop and finite-difference check) in seconds.
*Verification essential:* the float32 finite-difference check disagreed with autograd (0.000894 vs
0.000505). Blindly trusting either number would have been wrong; the fix was float64. Likewise,
reading "4/4 correct" from one seed would have hidden a 35–50% success rate for the 2-2-1 network
with Adam.

**7. Tests to keep when scaling up.**

- *Keep:* loss curves, prediction accuracy on held-out data, gradient-norm monitoring per layer,
  activation and pre-activation statistics (to detect dead or saturated units), multiple seeds for
  key claims, softmax sums and NaN checks, and a single spot-check of gradients on a tiny model.
- *Becomes too expensive:* exhaustive finite-difference gradient checks (2 forward passes per
  parameter, so billions of passes), per-example gradient loops, and tracing full weight matrices at
  every step. These are replaced by sampled checks and summary statistics.
