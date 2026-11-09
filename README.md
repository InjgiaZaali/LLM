# Cross-Lingual Multi-Label Emotion Classification (XED, English + Spanish)

Group 10. Prompting vs. parameter-efficient fine-tuning (QLoRA) on the XED
subtitle emotion dataset, English plus Spanish (resource band 5).
The original project brief is in `docs/project_brief.md`.

## Overview

- Task: multi-label classification of subtitle lines into 8 emotions
  (anger, anticipation, disgust, fear, joy, sadness, surprise, trust).
- Primary model: `Qwen/Qwen2.5-1.5B-Instruct`. Second model (prompting only):
  `ibm-granite/granite-3.3-2b-instruct`.
- Conditions: zero-shot, five-shot, English-only PEFT, Spanish-only PEFT,
  bilingual PEFT. Seed 42 for the split and for prompting.

## Repository layout

    configs/            model configuration (models.json)
    data/processed/     matched English-Spanish pairs and the train/dev/test TSVs
    scripts/            data preparation, split script, prompting notebooks
    src/peft/           PEFT code: formatting, training, generation, metrics
    predictions/        prediction files (dev/test) and metric CSVs
    run_condition.sh    SLURM job: train one PEFT condition, predict on dev
    run_test.sh         SLURM job: predict on test for all nine adapters
    requirements.txt    Python packages (see Environment)
    docs/               project brief

## Data and split

- `scripts/prepare_xed_dataset.py` builds the matched English-Spanish file.
- `scripts/split_en_es_dataset.py` creates the 70/20/10 split (seed 42,
  iterative multi-label stratification, grouped by English text so that
  alignment IDs and identical sentences never cross partitions; assertions
  stop the script if leakage is found).
- Sizes (rows / unique alignment IDs): train 3,349 / 2,664, dev 954 / 772,
  test 483 / 390. Some rows share an alignment ID, and at least 22 Spanish
  lines have corrupted accents (found with a regex check). Scores on clean
  lines equal the overall scores, so we treat this as a limitation.
- Spanish labels are projected from English, not independent human labels.

## Environment (Habrok cluster)

    module purge
    module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
    python -m venv --system-site-packages venv_llm
    source venv_llm/bin/activate
    pip install transformers peft trl bitsandbytes datasets accelerate scikit-learn "numpy<2" "pyarrow<20" "pandas<2.3" "scipy<1.15"
    export PYTHONPATH=$PWD/venv_llm/lib/python3.12/site-packages:$PYTHONPATH

`requirements.txt` is a `pip freeze` of this environment. It also lists packages
that come from the cluster module. Versions used: Python 3.12.3, PyTorch 2.7.1,
transformers 5.20.1, peft 0.21.2, trl 1.16.0, bitsandbytes 0.50.2,
scikit-learn 1.9.1. Paths in `run_condition.sh` and `run_test.sh` point to
`/scratch/s6866301`. Edit them for another machine.

## Reproducing the PEFT results

1. Format the data (writes `data/formatted/*.jsonl`, not committed):

        python src/peft/format_data.py
        cat data/formatted/dev_en.jsonl data/formatted/dev_es.jsonl > data/formatted/dev_bilingual.jsonl
        mkdir -p logs predictions

2. Train three conditions x three seeds, with dev predictions:

        for s in 42 43 44; do
          sbatch run_condition.sh english_1ep train_en $s dev_en
          sbatch run_condition.sh spanish_1ep train_es $s dev_es
          sbatch run_condition.sh bilingual_1ep train_bilingual $s dev_bilingual
        done

3. Test predictions, run once after the setup was frozen on dev:

        sbatch run_test.sh

4. Metrics (mean and sample std over seeds; use `dev` or `test`):

        python src/peft/full_metrics.py test
        python src/peft/error_analysis.py

## PEFT configuration

- QLoRA: 4-bit NF4, double quantization, bf16 compute.
- LoRA: all linear layers, rank 16, alpha 32, dropout 0.05
  (18,464,768 trainable parameters, 1.18%).
- Learning rate 2e-4, cosine schedule, warmup 3% of steps, per-device batch 8,
  gradient accumulation 2 (effective 16), max sequence length 256,
  1 epoch, seeds 42/43/44.
- Loss on the answer tokens only. Answers are JSON lists in fixed alphabetical
  order. The prompt wording is the same as in the prompting experiments.
- Epoch choice: in first runs with 3 epochs, dev loss was lowest after epoch 1
  in all conditions (English 0.449, 0.469, 0.592; Spanish 0.483, 0.493, 0.585;
  bilingual 0.475, 0.529, 0.729), so the final runs use 1 epoch.
  Checkpoint rule: lowest dev loss. Dev set per condition: English,
  Spanish, or both.
- Decoding: greedy, max 30 new tokens.
- Hardware: V100 or A100 on Habrok (SLURM partition gpushort, 1 GPU).
  Three-epoch runs on a V100 took 36, 38 and 70 minutes. One-epoch runs took
  about 5 to 7 minutes on an A100 and 16 to 22 minutes on a V100.
- Model weights and adapters are not committed (`outputs/` is ignored).

## Output parser

`src/peft/generate.py` parses the model output as a JSON list, lowercases and
strips labels, removes duplicates, and treats invalid output as an empty
prediction with a malformed flag. The parser was refactored after the runs:
re-parsing all 42 saved prediction files (31,590 predictions) with the new
parser gives identical predicted labels and malformed flags.

## Evaluation

Metrics are computed from the same binarized 8-label matrix in a fixed label
order with scikit-learn 1.9.1, `zero_division=0`. Primary: micro-F1 and
sample-averaged Jaccard. Secondary: macro-F1, micro precision and recall,
Hamming loss (lower is better). Diagnostic: per-emotion metrics, exact match,
malformed rate. Standard deviations use n-1 over three seeds.

## Prompting experiments

Zero-shot and five-shot prompting for Qwen and Granite are in
`scripts/Qwen_Zero_Clean.ipynb`, `scripts/Qwen_FiveShot_clean.ipynb` and
`scripts/Secondary_LLM_clean.ipynb` (Colab notebooks, fp16). Five few-shot
examples are drawn from the training set with `random_state=42`. Their metrics
come from the notebooks' own evaluation code, not from `src/peft`.

## Results

Test micro-F1, mean of 3 seeds for PEFT (single runs for prompting):

    Condition             English   Spanish
    Qwen zero-shot        0.336     0.310
    Qwen five-shot        0.343     0.293
    Granite five-shot     0.359     0.307
    English-only PEFT     0.464     0.412
    Spanish-only PEFT     0.419     0.382
    Bilingual PEFT        0.474     0.406

All metrics: `predictions/all_metrics_test_summary.csv` and
`predictions/all_metrics_dev_summary.csv`.

## Known limitations

- Prompting and PEFT metrics come from different code. Prompting results are
  single runs in fp16, PEFT used 4-bit QLoRA.
- Only the epoch count was selected on dev. Rank, alpha and learning rate are
  defaults, and checkpoints were chosen by dev loss, not dev F1.


### Prompting outputs and reproducibility

The three Colab notebooks in `scripts/` implement zero-shot and five-shot prompting with Qwen and Granite.

The five fixed training demonstrations are saved in `data/few_shot/qwen_five_shot_examples.tsv`.

- Development predictions: `predictions/prompting/dev/` (954 examples per experiment).
- Test predictions: `predictions/prompting/test/` (483 examples per experiment).
- Prompting evaluation metrics: `predictions/`, including `FINAL_LLM_TEST_COMPARISON.csv`.

To reproduce prompting experiments in Google Colab, enable GPU acceleration, install the dependencies listed in the notebooks, and provide the corresponding datasets from `data/processed/`. Five-shot experiments additionally require the fixed examples TSV. The notebooks currently use Google Drive paths under `/content/drive/MyDrive/`, so these paths must be configured before running.

Saved predictions include reference labels, parsed predictions, raw model outputs, and malformed-output flags. Final evaluation uses Micro-F1 and sample-averaged Jaccard as primary metrics.

Prompt-Wordings Sensitivity Analysis

To evaluate the effect of prompt wording, we conducted an additional controlled experiment using Qwen2.5-1.5B-Instruct on 200 paired English–Spanish examples sampled from the development set (seed 42). We compared the original zero-shot prompt with a modified instruction explicitly stating that zero, one, or multiple emotion labels could be returned. The model, examples, decoding strategy, and maximum generation length were kept constant.
### Prompt-wording sensitivity analysis

A controlled prompt-wording sensitivity experiment was conducted using Qwen2.5-1.5B-Instruct on 200 paired English–Spanish development examples (seed 42).

The original and explicit multi-label prompts were compared under identical generation settings.

| Language | Original Micro-F1 | Modified Micro-F1 |
|---|---:|---:|
| English | 0.3409 | 0.3363 |
| Spanish | 0.2869 | 0.2805 |

The original wording achieved slightly higher Micro-F1 in both languages.

- Script: `scripts/prompt_sensitivity_dev.py`
- Metrics: `results/prompt_sensitivity/qwen_dev_prompt_sensitivity_metrics.csv`
- Predictions: `results/prompt_sensitivity/qwen_dev_prompt_sensitivity_predictions.csv`

The held-out test set was not used for this sensitivity experiment.
