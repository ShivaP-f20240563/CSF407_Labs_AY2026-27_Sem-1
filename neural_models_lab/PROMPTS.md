# LLM Prompt Log (Neural Models Lab)

## Prompt 1: first implementation (Task 3)

> Generate minimal PyTorch code for the following model and dataset. Do not change the architecture
> or task.
> Dataset (XOR, full batch): X = [[0,0],[0,1],[1,0],[1,1]], y = [0,1,1,0].
> Model: 2 inputs → 2 hidden units with tanh → 1 output logit. Use BCEWithLogitsLoss. Random weight
> initialisation, torch.manual_seed(0), full-batch Adam, lr 0.05, 3000 steps on CPU.
> After training, report the final loss, all four probabilities, thresholded labels, and the gradient
> tensor of the first-layer weights after backward(). Set a random seed for reproducibility and
> explain each test in one sentence.

**Inspection before running** (where each step happens):

| Step | Code |
|---|---|
| forward pass | `logits = net(X)` |
| scalar loss | `loss = loss_fn(logits, target)` |
| reverse-mode AD | `loss.backward()` |
| parameter update | `optim.step()` |

**Changes made before execution:**

1. Confirmed that the gradient is read *after* `backward()` and *before* the next `zero_grad()`.
   Otherwise `.grad` would print zeros.
2. Made the activation a parameter (`make_net(act=...)`) so that Part D changes *only* the activation.

## Prompt 2: diagnostics

> Add: (a) a check that the batch gradient equals the mean of the 4 per-example gradients; (b) a
> central finite-difference check of one weight; (c) a copy of the experiment where every weight and
> bias is set to the same constant, printing both rows of W1 at steps 0, 1, 10, 100 and the last step.

**Correction:** the finite-difference check disagreed with autograd in float32 (0.000894 against
0.000505). The cause was round-off with eps = 1e-4, not a backprop bug. Switching to float64 made them
agree to 6 decimal places.

## Prompt 3: three-class extension (Task 5)

> Modify only the output and loss part of the previous code: 3 output logits, CrossEntropyLoss, targets
> [0, 1, 1, 2]. Print the softmax probabilities for all four inputs, check they sum to 1, check that
> adding 100 to all logits leaves softmax unchanged, and verify numerically that dL/dz = (p − y)/N.

## Where human verification was essential

- The zero-initialisation run printed rows that were identical *and still zero*. Explaining why
  (with the output weights also at 0, no gradient reaches W1 at all) required reasoning, and led to
  adding the constant-0.5 variant, which shows identical but *moving* rows.
- The success rate of a 2-2-1 network over repeated seeds was well below 100%. One seed is not
  evidence; see the repeated-runs table.
