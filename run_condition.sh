#!/bin/bash
#SBATCH --job-name=xed_peft
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log

# Usage: sbatch run_condition.sh <condition> <train_file> <seed>
#   e.g. sbatch run_condition.sh english train_en 42
COND=$1
TRAIN=$2
SEED=${3:-42}
DEV=${4:-dev_en}

source /scratch/s6866301/setup_env.sh
export HF_HUB_OFFLINE=1
cd /scratch/s6866301/LLM

OUT=outputs/${COND}_seed${SEED}
python src/peft/train_lora.py \
  --train data/formatted/${TRAIN}.jsonl \
  --dev data/formatted/${DEV}.jsonl \
  --out $OUT --seed $SEED

for LANG in en es; do
  python src/peft/generate.py --adapter $OUT/best_adapter \
    --data data/formatted/dev_${LANG}.jsonl \
    --condition $COND --seed $SEED \
    --out predictions/dev_${COND}_${LANG}_seed${SEED}.tsv
done
