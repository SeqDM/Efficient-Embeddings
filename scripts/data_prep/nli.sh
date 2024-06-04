#!/bin/bash

if ! [ -f 'data/downloaded/nli_for_simcse.csv' ]; then
    mkdir -p data/downloaded
    wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/nli_for_simcse.csv \
        --directory-prefix=data/downloaded
fi

mkdir -p data/nli
./scripts/data_prep/nli.py \
    data/downloaded/nli_for_simcse.csv \
    data/nli/all.jsonl \
