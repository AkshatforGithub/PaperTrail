#!/usr/bin/env bash
# Re-ingest the frozen corpus at several chunk sizes and evaluate each.
#   bash scripts/chunk_experiment.sh                    # uses the hybrid retriever
#   RETRIEVER=hybrid_rerank bash scripts/chunk_experiment.sh
# The default size (800/100) runs last, so the database ends in its normal state.
set -euo pipefail
RETRIEVER="${RETRIEVER:-hybrid}"
for cfg in "400 50" "1200 150" "800 100"; do
  set -- $cfg
  size=$1
  overlap=$2
  echo "=== chunk_size=$size overlap=$overlap retriever=$RETRIEVER ==="
  CHUNK_SIZE=$size CHUNK_OVERLAP=$overlap make ingest ARGS="--ids-file evaluation/corpus.txt --reingest"
  CHUNK_SIZE=$size CHUNK_OVERLAP=$overlap python evaluation/run_eval.py \
    --retriever "$RETRIEVER" --name "chunk${size}_${RETRIEVER}" --force
done
