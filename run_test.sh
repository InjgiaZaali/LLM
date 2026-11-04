#!/bin/bash
#SBATCH --job-name=xed_test
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log

source /scratch/s6866301/setup_env.sh
export HF_HUB_OFFLINE=1
cd /scratch/s6866301/LLM

for COND in english_1ep spanish_1ep bilingual_1ep; do
  for SEED in 42 43 44; do
    for LANG in en es; do
      python src/peft/generate.py \
        --adapter outputs/${COND}_seed${SEED}/best_adapter \
        --data data/formatted/test_${LANG}.jsonl \
        --condition $COND --seed $SEED \
        --out predictions/test_${COND}_${LANG}_seed${SEED}.tsv
    done
  done
done
