# Experiment journal: what we tried and what happened

This page records the **actual sequence**, including failed attempts. The path names below refer to the datasets in this repository. Adapters and model weights were local and are not published. The same 18 notes, three test notes, and WORK→R31 / PERSONAL→R58 / SHOPPING→R74 mapping were used throughout.

## Baseline

The downloaded base model could classify simple notes but had not been told our arbitrary internal route-code mapping. On the route-code prompt it spent its 256-token generation budget reasoning about possible codes and produced no final answer. For the final v3 prompt, it recognized `Call Mom on Sunday.` as personal in analysis but again produced no final answer within 256 tokens. This is why the exercise asks the adapter to learn a private mapping instead of only a common category.

## Attempts and observed results

| Attempt | Data and target | Adapted layers; steps; learning rate | What the recorded run showed |
| --- | --- | --- | --- |
| Smoke 1 | v1, `ROUTE: Rxx` | 4 layers; 2; `1e-4` | Backward pass and save worked on the Mac. This was not an accuracy test. |
| 1 | v1 | 4 layers; 60; `1e-4` | Held-out exact match **1/3**. SHOPPING→R74 was right; WORK and PERSONAL also emitted R74. |
| 2 | v1, continued from attempt 1 | 4 layers; **120 additional**; `1e-4` | A seen WORK training note still emitted R74. More updates alone did not fix the collapse. |
| Smoke 2 | v1 | 16 layers; 2; `1e-4` | Larger adapter trained without a reported memory error. |
| 3 | v1 | 16 layers; 60; `1e-4` | Held-out exact match **2/3**: WORK→R31 and SHOPPING→R74 right; PERSONAL→R31 wrong. A seen PERSONAL note was also wrong. |
| 4 | v2, two-line `CATEGORY` / `ROUTE` | 16 layers; 60; `1e-4` | A seen PERSONAL training note returned malformed `CATEGORY: R74`, even when retested with actual newline characters. We did not treat this as a useful adapter. |
| 5 | v3, compact `PERSONAL R58` style | 16 layers; 120; `1e-5` | Seen PERSONAL note correct. Held-out WORK, PERSONAL, and SHOPPING all correct: **3/3** exact in this run. |

The final v3 training loss reached 0.001, but the **generated answers** are the evidence of task performance. Earlier runs had low average loss while the distinguishing route code was wrong. For v1, inspection showed an answer suffix of nine tokens, **eight shared** across classes; only one code token differed. V3's answer suffix contained distinct category tokens as well as the code. This gives a plausible explanation, not proof of causation: the final run also used a smaller learning rate and more updates.

## Reproduce an earlier attempt manually

These are optional. Complete the successful path in the main [guide](../README.md) first. Each command starts from the frozen base unless `--resume-adapter-file` is present. Use a **new** `--adapter-path` when repeating an experiment so you can compare outputs.

### Attempt 1: four adapted layers, route only

```bash
mlx_lm.lora \
  --model models/gpt-oss-20b-mlx --train --data data/routes_v1 \
  --adapter-path outputs/routes-v1-4layer \
  --fine-tune-type lora --num-layers 4 --batch-size 1 --iters 60 \
  --max-seq-length 256 --learning-rate 1e-4 \
  --mask-prompt --grad-checkpoint --steps-per-report 10
```

**Why:** This is the initial design. The 4-layer adapter changed only about **0.088%** of model parameters (18.451 million of 20.915 billion), but it mostly learned to emit the output shape and one frequent code. To compare on a test note, use the prompt helper with `--version routes_v1`, and set `--adapter-path outputs/routes-v1-4layer` in `mlx_lm.generate`.

### Attempt 2: continue the same adapter

```bash
mlx_lm.lora \
  --model models/gpt-oss-20b-mlx --train --data data/routes_v1 \
  --resume-adapter-file outputs/routes-v1-4layer/adapters.safetensors \
  --adapter-path outputs/routes-v1-4layer-continued \
  --fine-tune-type lora --num-layers 4 --batch-size 1 --iters 120 \
  --max-seq-length 256 --learning-rate 1e-4 \
  --mask-prompt --grad-checkpoint --steps-per-report 20
```

**Why:** `--resume-adapter-file` loads the learned adapter weights before further training; `--adapter-path` keeps the continued result separate. The recorded run still answered a seen WORK note with R74, so more steps alone were not enough. MLX-LM's [LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) documents this resume option.

### Attempt 3: increase adapted capacity to 16 layers

```bash
mlx_lm.lora \
  --model models/gpt-oss-20b-mlx --train --data data/routes_v1 \
  --adapter-path outputs/routes-v1-16layer \
  --fine-tune-type lora --num-layers 16 --batch-size 1 --iters 60 \
  --max-seq-length 256 --learning-rate 1e-4 \
  --mask-prompt --grad-checkpoint --steps-per-report 10
```

**Why:** With 16 layers, the adapter had about **73.806 million trainable parameters (0.353%)**. WORK and SHOPPING test notes became correct, but PERSONAL remained wrong, including a training note. More adapter capacity helped in this run but did not solve the whole task.

### Attempt 4: expose category and route in two lines

```bash
mlx_lm.lora \
  --model models/gpt-oss-20b-mlx --train --data data/routes_v2 \
  --adapter-path outputs/routes-v2-16layer \
  --fine-tune-type lora --num-layers 16 --batch-size 1 --iters 60 \
  --max-seq-length 256 --learning-rate 1e-4 \
  --mask-prompt --grad-checkpoint --steps-per-report 10
```

**Why:** This changed the target to two lines so errors in category and code would be visible separately. In the recorded run, a seen PERSONAL prompt got `CATEGORY: R74`, putting a route code in the category field. That is a failure even though training completed. We verified that a real newline in the generation prompt did not fix it.

### Attempt 5: compact target and smaller updates

Use step 8 in the main [guide](../README.md), which trains `data/routes_v3` for **120 updates** at **`1e-5`** with 16 adapted layers. Both the category and code now vary among the supervised answer tokens. The three development test notes were correct in the recorded run.

## What this result can and cannot say

The final adapter learned the requested behavior on these examples. It was **not** tested as a general-purpose classifier. The three test notes were checked after multiple designs, and their failures informed later changes. They were held out from **weight updates**, but they were **not held out from experiment design**. For a credible generalization estimate, collect fresh notes after fixing the design, keep them unseen until the end, and use more than one training run. Because v3 changed output format, learning rate, and step count together, this journal cannot isolate the effect of any single change.
