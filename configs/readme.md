# Model Selection

The project uses two pretrained instruction-tuned language models:

**Primary model: Qwen2.5-1.5B-Instruct**
- Approximately 1.5 billion parameters.
- Used for zero-shot and five-shot prompting.
- Also used for LoRA/QLoRA fine-tuning.

**Secondary model: Granite-3.3-2B-Instruct**
- Approximately 2 billion parameters.
- Used for zero-shot and five-shot prompting only.

Both models support multilingual tasks, including English and Spanish, and are distributed under Apache 2.0 licenses.

Their relatively small sizes make them suitable for experimentation using available GPU resources.

The project uses random seed 42 for the fixed five-shot demonstration selection. PEFT experiments also include seeds 42, 43, and 44.
