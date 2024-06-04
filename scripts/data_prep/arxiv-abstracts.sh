#!/bin/bash

if ! [ -f 'data/downloaded/arxiv-metadata-oai-snapshot.json.zip' ]; then
    echo Please, log in to Kaggle and download the data from
    echo https://www.kaggle.com/datasets/Cornell-University/arxiv/data
    echo It should be saved to data/downloaded arxiv-metadata-oai-snapshot.json.zip
    mkdir -p data/downloaded
else
    if ! [ -f 'data/downloaded/arxiv-metadata-oai-snapshot.json' ]; then
        cd data/downloaded
        unzip arxiv-metadata-oai-snapshot.json.zip
        cd ../..
    fi
    mkdir -p data/arxiv-abstracts
    ./scripts/data_prep/arxiv-abstracts.py \
        data/downloaded/arxiv-metadata-oai-snapshot.json \
        data/arxiv-abstracts/all.jsonl
fi
