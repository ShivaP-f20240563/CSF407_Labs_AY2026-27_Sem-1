# Neural Models Lab: Problem Specification and Model Design

## Task 1: The Problem Before Any Code

**Input space:** 𝒳 = {0,1}², the two binary sensors (x₁, x₂).
**Output space:** 𝒴 = {0,1}, where 1 means "sensor-disagreement warning".

| x₁ | x₂ | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Sketch:**

```
 x2
 1 |  (0,1)=1 ●        ○ (1,1)=0
   |
 0 |  (0,0)=0 ○        ● (1,0)=1
   +------------------------- x1
        0              1
```

**Why one straight line fails:** the positive points (0,1) and (1,0) lie on one diagonal and the
negative points (0,0) and (1,1) on the other. Any line that puts both positives on one side also puts
at least one negative there. Algebraically, w₂ + b > 0 and w₁ + b > 0 imply w₁ + w₂ + 2b > 0. But
b < 0 and w₁ + w₂ + b < 0 together give w₁ + w₂ + 2b < 0, a contradiction.

**Prediction for the linear model:** a single affine map followed by a sigmoid cannot fit the data.
By symmetry, the best it can do is output p = 0.5 for every input, so the loss will plateau at
ln 2 ≈ 0.693 and at most 2 of 4 points will be classified correctly.
*(Confirmed in section 0 of `xor_lab.py`: loss 0.6931, all probabilities 0.5000.)*

**Think about it:** XOR tests the claim that *having the right kind of representation matters more
than having many parameters*. The linear model's failure comes from its function class, not from
too little training. The 2-2-1 network succeeds because its hidden layer can *re-represent* the
inputs so that the classes become linearly separable.

## Task 2: Model Design (on paper, before prompting)

```
x ∈ ℝ²  →  Linear(2→2)  →  tanh (later sigmoid, ReLU)  →  Linear(2→1)  →  logit z
                                                                           ↓
                                             p = σ(z),  loss = BCE(p, y)  (BCEWithLogitsLoss)
optimiser: full-batch Adam, lr = 0.05, 3000 steps, fixed seed
```

**1. Why the hidden nonlinearity is scientifically necessary:** without it,
W⁽²⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾ = W′x + b′ is still a single affine map. Extra depth adds no expressive
power, and XOR stays impossible. A nonlinearity bends the input space so the hidden layer can compute
features such as "OR" and "AND", from which XOR = OR ∧ ¬AND is linearly separable.

**2. Why sigmoid + binary cross-entropy is a sensible engineering pairing:** the sigmoid maps a logit
to a Bernoulli probability, and BCE is the negative log-likelihood of that Bernoulli. Together the
gradient with respect to the logit is simply p − y. The sigmoid's saturation cancels out, so a
confidently wrong output still gets a strong gradient. `BCEWithLogitsLoss` fuses the two for
numerical stability.

**3. Validation criteria (evidence of successful learning):**

1. The final loss is far below ln 2 (target < 0.01).
2. All four thresholded predictions (p > 0.5) equal the XOR labels.
3. The first-layer gradients are non-zero early in training and match a finite-difference estimate.
4. Repeated runs over several seeds report the success rate, not a single lucky run.
5. The hidden-layer weight rows become *different*, showing that the two units learned distinct features.

**Think about it:** the hidden units have no targets of their own. What each one computes is decided
entirely by backpropagation. The output error is sent backwards through W⁽²⁾ and the activation
derivative, and each hidden unit is pushed in whichever direction most reduces the *final* loss.
Hidden features emerge as a by-product of minimising the output loss.
