#!/bin/bash

set -e

# Run this from the root directory of this repository!

GDRIVE_ID='1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N'
DATA_DIR='raw_data/NYT-CopyRE'
PREPREPROCESS_SCRIPT='raw_data.NYT-CopyRE.prepreprocess'

# 1. Create a temporary directory.
TMP_DIR=`mktemp --directory`

# 2. Download the data from Google Drive.
TMP_ZIP="${TMP_DIR}/raw_nyt.zip"
gdown "${GDRIVE_ID}" --output "${TMP_ZIP}"

# 3. Unzip the downloaded data.
TMP_UNZIP="${TMP_DIR}/raw_nyt"
unzip "${TMP_ZIP}" -d "${TMP_UNZIP}"

# 4. Pre-preprocess each of the unpacked files.
python -m "${PREPREPROCESS_SCRIPT}" \
    --source="${TMP_UNZIP}/raw_train.json" \
    --target="${DATA_DIR}/train.jsonl"
python -m "${PREPREPROCESS_SCRIPT}" \
    --source="${TMP_UNZIP}/raw_valid.json" \
    --target="${DATA_DIR}/validation.jsonl"
python -m "${PREPREPROCESS_SCRIPT}" \
    --source="${TMP_UNZIP}/raw_test.json" \
    --target="${DATA_DIR}/test.jsonl"

# 5. Clean up the temporary directory.
rm -vrf "${TMP_DIR}"
