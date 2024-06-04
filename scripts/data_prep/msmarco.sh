#!/bin/bash

if ! [ -f 'data/downloaded/msmarco/train_v2.1.json' ]; then
    mkdir -p data/downloaded/msmarco
#    wget https://msmarco.blob.core.windows.net/msmarco/eval_v2.1_public.json.gz \
#        --directory-prefix=data/downloaded/msmarco
    wget https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz \
        --directory-prefix=data/downloaded/msmarco
    wget https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz \
        --directory-prefix=data/downloaded/msmarco

#    gzip -d data/downloaded/msmarco/eval_v2.1_public.json.gz
    gzip -d data/downloaded/msmarco/dev_v2.1.json.gz
    gzip -d data/downloaded/msmarco/train_v2.1.json.gz
fi

mkdir -p data/msmarco
#./scripts/data_prep/msmarco.py \
#    data/downloaded/msmarco/eval_v2.1_public.json \
#    data/msmarco/test.jsonl
./scripts/data_prep/msmarco.py \
    data/downloaded/msmarco/dev_v2.1.json \
    data/msmarco/dev.jsonl
./scripts/data_prep/msmarco.py \
    data/downloaded/msmarco/train_v2.1.json \
    data/msmarco/train.jsonl

echo Generated data saved to data/msmarco/train.jsonl and data/msmarco/dev.jsonl
