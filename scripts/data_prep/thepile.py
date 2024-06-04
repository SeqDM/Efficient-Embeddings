#!/bin/env python3

import sys
import json

from sample_positive_pairs import random_crop, random_drop

categories = [
    'Wikipedia (en)',
    'ArXiv',
    'PhilPapers',
    'Pile-CC',
] # TODO maybe more categories?

n_random_pairs = 5

data_in = sys.argv[1]
data_out = sys.argv[2]

def prep(text : str):
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    text = text.strip()
    text = text.split(' ')
    return text

batch_size = 1000000 # write every 1 mil. examples
with open(data_in, 'r') as f, open(data_out, 'w') as f_out:
    pairs = []
    for l in f:
        j = json.loads(l)
        text = j['text']
        category = j['meta']['pile_set_name']
        if category in categories:
            for _ in range(n_random_pairs):
                x = ' '.join(random_drop(random_crop(prep(text))))
                y = ' '.join(random_drop(random_crop(prep(text))))
                positive_example = json.dumps({'query': x, 'pos': y})
                pairs.append(positive_example)
        if len(pairs) > batch_size:
            print('.', end='', flush=True)
            f_out.write('\n'.join(pairs) + '\n')
            pairs = []
    f_out.write('\n'.join(pairs) + '\n')
    print()
