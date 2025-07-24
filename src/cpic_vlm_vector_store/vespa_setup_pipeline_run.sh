#!/usr/bin/env bash
set -e

MODEL_NAME="vidore/colqwen2.5-v0.2"
CACHE_DIR="/home/jovyan/datasets/cc-20250630151645/"
DEVICE="cuda:0"
BATCH=4
CPIC_DIR="/home/jovyan/datasets/cc-20250630151645/src/Guidelines"
FEED_JSON="vespa_feed.json"

# --- Vespa ---
DEPLOY=true           # set to false if you only want the feed JSON
TENANT="hsinnosukejp"
APP_NAME="cpicguidelinevrag"
VESPA_KEY=""

# ----- build optional flag ---------------------------------------------------
DEPLOY_FLAG=""
if [ "$DEPLOY" = true ]; then
  DEPLOY_FLAG="--deploy-vespa"
fi

# ----- execute pipeline ------------------------------------------------------
python vespa_setup_pipeline.py \
  --model-name      "$MODEL_NAME" \
  --cache-dir       "$CACHE_DIR" \
  --device          "$DEVICE" \
  --batch-size      "$BATCH" \
  --cpic-dir        "$CPIC_DIR" \
  --feed-output     "$FEED_JSON" \
  $DEPLOY_FLAG \
  --tenant-name     "$TENANT" \
  --vespa-app-name  "$APP_NAME" \
  --vespa-key       "$VESPA_KEY"
