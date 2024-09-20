#!/bin/bash

set -e

new_exps=(
    # redocred_seq2umt_pso
    redocred_large_seq2umt_pso
)

for exp in "${new_exps[@]}"; do
	python main.py -e $exp -m preprocessing
    python main.py -e $exp -m train
done
