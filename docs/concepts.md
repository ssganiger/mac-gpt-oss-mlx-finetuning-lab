# Concepts in this exercise

| Term | Meaning here |
| --- | --- |
| Base model | The downloaded gpt-oss-20b weights before this route-code training. |
| Quantization | Storing many weights in fewer bits to reduce the model's memory and disk needs. This lab uses the `MXFP4-Q8` MLX copy. |
| MLX | Apple's array and compute framework used by MLX-LM on Apple Silicon. |
| Unified memory | Memory shared by CPU and GPU on Apple Silicon. It still has a finite capacity; model weights, activations, adapters, and other apps all use it. |
| Tokenizer | Turns text and special chat markers into token IDs and back. A word may use one or several tokens. |
| Chat template | Adds structural markers for user and assistant messages. The GPT-OSS template includes an assistant `final` channel. |
| Training example | One user question paired with the assistant answer we want. |
| Training split | The 18 notes whose answers are used to calculate gradients and update adapters. |
| Test split | Three different notes not used for weight updates. We inspected them repeatedly while changing the experiment, so they are a development test set, not an independent final benchmark. |
| Supervised fine-tuning | Teaching a model to match example answers. Here we use the `mlx_lm.lora` command. |
| Forward pass | The model predicts the next answer tokens using its current weights and adapters. |
| Loss | A number measuring mismatch between those predictions and the desired answer tokens. Low loss on training batches alone does not guarantee correct generation. |
| Backpropagation and gradient | Calculations showing how trainable adapter weights should change to reduce loss. |
| Optimizer and learning rate | The optimizer applies updates; the learning rate controls their size. The successful recorded run used Adam at `1e-5`. |
| LoRA adapter | Small trainable weight matrices attached to selected model layers. The quantized base stays frozen. MLX-LM calls LoRA on a quantized model QLoRA. |
| Rank | Adapter capacity setting. MLX-LM's default rank in the recorded version was 8. |
| Layer | A processing block in the model. We adapted 4 layers in early runs and 16 in later runs. |
| Batch size | How many examples participate in one update; this lab uses 1. |
| Iteration or step | One optimizer update. With batch size 1, 120 steps read about 120 examples, sampling across 18 training records. |
| Epoch | One approximate pass through all training records; 120 steps is about 6⅔ passes with batch size 1. |
| Prompt masking | Excludes user-question tokens from training loss so updates focus on the assistant answer. |
| Gradient checkpointing | Saves memory during training by recomputing some intermediate results. |
| Inference | Generating an answer using the base model, optionally with the trained adapter. No weight update occurs. |
| Exact match | The generated final answer equals the desired text exactly, including format and code. |

## The three versions of the answer

| Version | Answer to a personal note | What changed |
| --- | --- | --- |
| v1 | `ROUTE: R58` | Only the route number varied within the answer-side tokens. |
| v2 | `CATEGORY: PERSONAL` followed by `ROUTE: R58` | Added the category, but also more shared formatting. |
| v3 | `PERSONAL R58` | Compact output with distinct category and code tokens. |

The categories and code mapping were unchanged across versions. The prompt wording changed to request each version's output shape. The final v3 run also changed the learning rate and update count, so the dataset format alone cannot be credited for its result.
