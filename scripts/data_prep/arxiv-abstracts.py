#!/bin/env python3

import sys
import json

data_in = sys.argv[1]
data_out = sys.argv[2]

def prep_abstract(a : str):
    a = a.replace('\n', ' ')
    a = a.replace('  ', ' ')
    a = a.strip()
    return a

with open(data_in) as f:
    data_in = json.loads('[' + ','.join(f.read().splitlines()) + ']')

pairs = []
for p in data_in:
    x = p['title']
    y = p['abstract']
    y = prep_abstract(y)
    positive_example = json.dumps({'query': x, 'pos': y})
    pairs.append(positive_example)

with open(data_out, 'w') as f:
    f.write('\n'.join(pairs) + '\n')
