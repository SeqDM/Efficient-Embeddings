#!/bin/bash

if ! [ -d 'data/downloaded/snli_1.0' ]; then
    mkdir -p data/downloaded
    cd data/downloaded
    wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip snli_1.0.zip
    cd ../..
fi

mkdir -p data/snli
./scripts/data_prep/snli.py data/downloaded/snli_1.0/snli_1.0_train.jsonl data/snli/train.jsonl
./scripts/data_prep/snli.py data/downloaded/snli_1.0/snli_1.0_dev.jsonl data/snli/dev.jsonl
./scripts/data_prep/snli.py data/downloaded/snli_1.0/snli_1.0_test.jsonl data/snli/test.jsonl
