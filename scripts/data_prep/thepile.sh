#!/bin/bash

STORAGE=/net/pr2/projects/plgrid/plggeffemb/downloaded/thepile
mkdir -p $STORAGE
mkdir -p data/thepile/{train,valid,test}

if ! [ -f "$STORAGE/29.jsonl.zst" ]; then
    for i in `seq -w 0 29`; do
        wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/$i.jsonl.zst \
        --directory-prefix=$STORAGE
    done
fi

if ! [ -f "$STORAGE/test.jsonl.zst" ]; then
    wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/test.jsonl.zst \
    --directory-prefix=$STORAGE
    unzstd $STORAGE/test.jsonl.zst
    echo Processing $STORAGE/test.jsonl ...
    ./scripts/data_prep/thepile.py \
        $STORAGE/test.jsonl \
        data/thepile/test.jsonl
    echo Processed data saved to data/thepile/test.jsonl
fi

if ! [ -f "$STORAGE/val.jsonl.zst" ]; then
    wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/val.jsonl.zst \
    --directory-prefix=$STORAGE
    unzstd $STORAGE/val.jsonl.zst
    echo Processing $STORAGE/val.jsonl ...
    ./scripts/data_prep/thepile.py \
        $STORAGE/val.jsonl \
        data/thepile/dev.jsonl
    echo Processed data saved to data/thepile/dev.jsonl
fi

if ! [ -f "$STORAGE/29.jsonl" ]; then
    for i in `seq -w 0 29`; do
        unzstd $STORAGE/$i.jsonl.zst
    done
fi

if ! [ -f "data/thepile/train/29.jsonl" ]; then
    for i in `seq -w 0 29`; do
        echo Processing $STORAGE/$i.jsonl ...
        ./scripts/data_prep/thepile.py \
            $STORAGE/$i.jsonl \
            data/thepile/train/$i.jsonl
        echo Processed data saved to data/thepile/train/$i.jsonl
    done
fi
