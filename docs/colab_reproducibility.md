# Google Colab and prompting reproducibility

The final prompting code is in:

- `scripts/Qwen_Zero_Clean.ipynb` (Qwen development zero-shot)
- `scripts/Qwen_FiveShot_clean.ipynb` (Qwen five-shot and Qwen final test conditions)
- `scripts/Secondary_LLM_clean.ipynb` (Granite development and final test conditions)

## Required input files

Download from GitHub and make available in the Colab `/content` directory:

- `data/processed/xed_train.tsv` as `/content/xed_train.tsv`
- `data/processed/xed_dev.tsv` as `/content/xed_dev.tsv`
- `data/processed/xed_test.tsv` as `/content/xed_test.tsv`

The existing notebooks load and save outputs from `/content/drive/MyDrive/`.
Mount Google Drive in Colab or edit those paths to another writable directory.
For the five-shot notebooks, also place
`data/few_shot/qwen_five_shot_examples.tsv` at
`/content/drive/MyDrive/qwen_five_shot_examples.tsv` before running.
This exact TSV is the experiment's fixed five-example sample and must not be
regenerated from a different train-set order.

Enable an NVIDIA GPU runtime. Run notebook cells in order. The Colab notebooks
install Python packages in their initial cells; note that running against newer
package versions can change behavior. The Qwen zero-shot notebook loads
`dtype="auto"`, while the other two notebooks explicitly use `torch.float16`.

## Saved outputs

- `predictions/prompting/dev/` — 8 prompting dev CSVs, 954 records per condition.
- `predictions/prompting/test/` — 8 prompting test CSVs, 483 records per condition.
- `predictions/FINAL_LLM_TEST_COMPARISON.csv` — final 8-condition test metrics.
- Other prompting evaluation CSVs are in `predictions/`.
- PEFT test and dev predictions (seed 42, 43, 44) and summary metrics are also
  kept in `predictions/`.

Parsed predictions use eight English emotion names. Invalid JSON or disallowed
labels are counted as malformed and mapped to an empty prediction. Metrics
are computed with `zero_division=0`.

## Controlled prompt-wording sensitivity (development only)

A separate *new* script, `scripts/prompt_sensitivity_dev.py`, compares the
original Qwen zero-shot wording against a wording that explicitly states that
one, several, or no emotions may be returned. The model, dataset subset,
decoding, token budget and output parser are held fixed. From the repository
root on a GPU-equipped Python environment run:

```bash
python scripts/prompt_sensitivity_dev.py --n 200 --seed 42
```

The script creates two files under `results/prompt_sensitivity/` **only once
executed**. Upload those actual CSVs and update the report with the observed
results and the fact that a 200-pair development subset was used. For a full
954-pair comparison use `--n 0`. **Do not use the held-out test set for prompt
selection.** Neither uploading the script nor the pre-existing prompting CSVs
counts as a completed prompt-wording sensitivity experiment.
