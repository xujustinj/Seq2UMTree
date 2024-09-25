#!/bin/bash

set -e

python main.py -e temp_nyt -m preprocessing
python main.py -e temp_nyt -m train
