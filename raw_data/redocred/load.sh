#!/bin/bash

DATA_DIR='raw_data/redocred/data'
GITHUB_URL='https://raw.githubusercontent.com/tonytan48/Re-DocRED/d62d5ad95850d26ab737eaf7a92448ebff68c816'

# 1. Make the data directory.
mkdir --parent "$DATA_DIR"

# 2. Download the data from GitHub.
for FILE in 'dev_revised.json' 'test_revised.json' 'train_revised.json'
do
    wget --output-document="$DATA_DIR/$FILE" "$GITHUB_URL/data/$FILE"
done

# 3. Run preprocessing.
python raw_data/redocred/preprocessing.py | tee raw_data/redocred/preprocessing.txt
