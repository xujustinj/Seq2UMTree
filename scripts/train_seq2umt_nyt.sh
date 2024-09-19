#!/bin/bash

set -e

new_exps=(
    # nyt_seq2umt_ops
    # nyt_seq2umt_osp
    # nyt_seq2umt_spo
    # nyt_seq2umt_sop
    nyt_seq2umt_pso
    # nyt_seq2umt_pos
)

for exp in "${new_exps[@]}"; do
	python main.py -e $exp -m preprocessing
    python main.py -e $exp -m train
done
