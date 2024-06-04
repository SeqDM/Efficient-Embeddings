# Efficient embeddings

Experiments for efficient repurposing embedding models via contrastive learning
of pre-trained LLMs.


## Requirements

* Python 3.10 -- 3.11

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Downloading and preprocessing data

The shell scripts for downloading and light pre-processing of various datasets
are located in `scripts/data_prep`. E.g., to download NLI dataset run:

```
./scripts/data_prep/nli.sh
```
and the dataset in JSONL file will be save to `data/nli/all.jsonl`.

The data used for experiments in the paper (the BAAI data set) can be downloaded
from https://data.baai.ac.cn/details/BAAI-MTP (login required).


## Running training and evaluation

To run training and MTEB evaluation for the four methods considered in the paper
(full fine-tuning, layer freezing, only-bias tuning, and LoRA), please run
an appropriate script from among `experiments/*.py`. For instance:

```
./experiments/train_and_eval_freezing.py
```

The checkpoints of the model and other data will be saved to the `result` directory.

`experiments/*.gin` files associated with the Python scripts define
configurations of the experiments (learning rate, weight decay, LoRA rank, etc.)
You can modify them.

In the file `data/mteb/test_tasks_list` there is a short list of MTEB tasks used
for evaluation for debugging purposes. The list used for evaluation in the paper
is in `data/mteb/broad_tasks_list`. (The former list is specified in the config
files.)
