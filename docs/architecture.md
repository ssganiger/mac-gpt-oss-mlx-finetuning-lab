# Mermaid maps: gpt-oss-20b + MLX-LM

These diagrams use the actual route-code exercise. Green shows data preparation, blue shows the model and software, orange shows training updates, purple shows results, and red shows Mac hardware. The [experiment journal](experiments.md) has the commands and observed outputs.

## 1. Complete v3 training and evaluation path

```mermaid
graph TD
    A["data/routes_v3/train.jsonl<br/>18 notes, 6 per category"] --> B["Chat template and tokenizer<br/>user prompt plus target answer"]
    B --> C["Prompt mask<br/>train on assistant answer tokens"]
    C --> D["MLX-LM forward pass<br/>predict WORK R31, PERSONAL R58, or SHOPPING R74"]
    M["Quantized gpt-oss-20b<br/>frozen base weights"] --> D
    L["LoRA adapters<br/>16 layers, rank 8"] --> D
    D --> E["Answer-token loss<br/>prediction versus desired label"]
    E --> F["MLX backpropagation<br/>calculate adapter gradients"]
    F --> G["Adam update<br/>120 steps at learning rate 1e-5"]
    G --> L
    L --> H["Save adapters.safetensors<br/>73.806M trainable parameters"]
    T["data/routes_v3/test.jsonl<br/>3 notes excluded from weight updates"] --> V["Generate with exact stored prompts<br/>compare base and adapted final answers"]
    M --> V
    H --> V
    V --> R["Recorded development test<br/>adapted 3/3 exact"]

    classDef data fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef model fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef train fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef result fill:#F3E8FF,stroke:#9333EA,color:#581C87,stroke-width:2px;
    class A,B,C,T data;
    class D,M,L model;
    class E,F,G train;
    class H,V,R result;
```

The base model could identify a personal note in its analysis but did not produce a final route code within 256 tokens. The final adapter answered all three development test notes in the requested format. Those notes were checked during earlier attempts, so 3/3 is a learning result, not an independent generalization estimate.

## 2. Why the exercise required several runs

```mermaid
graph LR
    BASE["Base model<br/>private route mapping unknown"] --> V1["v1 target: ROUTE: Rxx<br/>4-layer LoRA, 60 steps"]
    V1 --> O1["1/3 test<br/>R74 repeated"]
    O1 --> EXT["Continue 4-layer adapter<br/>120 more steps"]
    EXT --> O2["Seen WORK note still wrong<br/>more steps alone failed"]
    O2 --> BIG["Fresh v1 adapter<br/>16 layers, 60 steps"]
    BIG --> O3["2/3 test<br/>PERSONAL still wrong"]
    O3 --> V2["v2 two-line answer<br/>16 layers, 60 steps"]
    V2 --> O4["Seen PERSONAL note malformed<br/>CATEGORY: R74"]
    O4 --> V3["v3 compact CATEGORY CODE<br/>16 layers, 120 steps, lr 1e-5"]
    V3 --> O5["3/3 development test<br/>WORK R31, PERSONAL R58, SHOPPING R74"]
    O5 --> CAUTION["Format, learning rate, and steps changed together<br/>cause not isolated"]

    classDef data fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef model fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef train fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef result fill:#F3E8FF,stroke:#9333EA,color:#581C87,stroke-width:2px;
    class BASE,V1,BIG,V2,V3 model;
    class EXT train;
    class O1,O2,O3,O4,O5 result;
    class CAUTION data;
```

The v1 answer had nine supervised tokens, eight shared across all categories. V3 made both category and code vary in the compact answer. That is a plausible reason for improvement; this run did not isolate it from the smaller learning rate or longer training.

## 3. Software, files, and unified memory on the Mac

```mermaid
graph LR
    subgraph DISK["Local files on disk"]
        HF["Hugging Face MLX model copy<br/>about 12.1 GB"]
        DATA["JSONL train and test files<br/>small, tracked in Git"]
        AD["Saved LoRA adapter<br/>about 282 MB in recorded run"]
    end

    subgraph STACK["Python software"]
        TOK["Tokenizer and chat template<br/>read model files"]
        LM["MLX-LM<br/>generation and LoRA training"]
        MX["MLX<br/>Apple Silicon compute"]
    end

    subgraph MAC["Recorded M3 Pro MacBook Pro"]
        UM["36 GB unified memory<br/>shared CPU and GPU physical pool"]
        Q["Frozen quantized model weights<br/>about 20.915B parameters"]
        L["Trainable LoRA weights<br/>73.806M, about 0.353%"]
    end

    HF --> TOK --> LM --> MX --> UM
    DATA --> LM
    UM --> Q
    UM --> L
    Q --> LM
    L --> LM
    L --> AD

    classDef file fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef framework fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef adapter fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef hardware fill:#FEE2E2,stroke:#C74634,color:#7F1D1D,stroke-width:3px;
    class HF,DATA,AD file;
    class TOK,LM,MX,Q framework;
    class L adapter;
    class UM hardware;
```

MLX uses Apple Silicon's shared physical memory pool. It still consumes memory for weights, adapters, activations, and other processes. The roughly 12.6 GB peak reported by **generation** is not a training-memory measurement. The 36 GB Mac is the tested configuration, not a proven minimum.
