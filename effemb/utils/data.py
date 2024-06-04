import json
import os
import random
from dataclasses import dataclass

import gin
import torch
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer


# Not used currently (causes incompatibilities with the accelerate library)
#@dataclass
#class Batch:
#    query: str
#    pos: str
#    neg: list[str]

# Calculated for when we limit the length by 75
AVERAGE_BAAI_QUERY_TOKENS = 14
AVERAGE_BAAI_POS_TOKENS = 64

class Tokenizer:
    def __init__(self, tokenizer_path, length=75):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.length = length

    def __call__(self, batch_text):
        batch_tensor = self.tokenizer(
            batch_text,
            max_length=self.length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        return batch_tensor


class IterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, datapath, offset=0):
        super(IterableDataset).__init__()
        self.datapath = datapath
        self.file = open(self.datapath, 'rb')
        self.offset = offset

    def __iter__(self):
        self.file.seek(self.offset)
        return self.file

    def current_position(self):
        return self.file.tell()


class IterableMixedDataset(torch.utils.data.IterableDataset):
    """
    This class defines dataset being a mixture of data from several files.
    The data sources and their proportions need to be specified like this:
        [('data/path/a', 8), ('data/path/b', 4)]
    When we reach the end of one of the files, this file will be silently
    read one more time from the beginning (thus the dataset is "infinite").
    """

    def __init__(self, datapaths_frequencies):
        super(IterableDataset).__init__()
        self.datapaths, self.frequencies = zip(*datapaths_frequencies)
        self.dataset_name = '__+__'.join(f'{d}__{f}' \
                                     for d, f in datapaths_frequencies)

    def __iter__(self):
        files = [open(d) for d in self.datapaths]
        current = 0
        while True:
            current = (current + 1) % len(files)
            file, freq = files[current], self.frequencies[current]
            for _ in range(freq):
                try:
                    yield next(file)
                except StopIteration:
                    # Start over
                    file.seek(0)
                    yield next(file)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=-1, tokenizer=None, use_neg=True):
        self.tokenizer = tokenizer
        self.use_neg = use_neg
        super(DataLoader, self).__init__(dataset, batch_size=batch_size,
                                         collate_fn=self.collate)

    def collate(self, examples):
        '''
        Making a batch out of a list of examples (lines from the data file(s)).'
        All the texts will be tokenized and put into PyTorch tensors.
        '''
        batch = {'query': [], 'pos': [], 'neg': []}
        for example in examples:
            example = json.loads(example)
            for k, v in example.items():
                batch[k].append(v)
        # E.g. in BAAI positives are lists of length 1, here we handle it
        if isinstance(batch['pos'][0], list):
            batch['pos'] = [i[0] for i in batch['pos']]
        batch['pos'] = self.tokenizer(batch['pos'])
        batch['query'] = self.tokenizer(batch['query'])
        if self.use_neg:
            # Datasets may contain arbitrary number of negatives, therefore lists
            if isinstance(batch['neg'][0], str):
                batch['neg'] = [[i] for i in batch['neg']]
            # We filter out empty negatives (they appear in BAAI sometimes)
            for i in range(len(batch['neg'])):
                batch['neg'][i] = [n for n in batch['neg'][i] if n]
                assert batch['neg'][i], \
                    "There are no negatives associated with the query"
            batch['neg'] = [self.tokenizer(n) for n in batch['neg']]
        else:
            batch.pop('neg')
        return batch


@gin.configurable
def get_dataloader(datapath=None, tokenizer=None, batch_size=-1, offset=0,
                   use_neg=False):
    extension = datapath.split(".")[-1]
    assert extension == "jsonl", "All datasets should be provided in JSON lines format"
    dataset = IterableDataset(datapath, offset=offset)
    return DataLoader(dataset, batch_size=batch_size, tokenizer=tokenizer,
                      use_neg=use_neg)
