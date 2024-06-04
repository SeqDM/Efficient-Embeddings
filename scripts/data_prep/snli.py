#!/bin/env python3

import sys
import json

data_in = sys.argv[1]
data_out = sys.argv[2]

with open(data_in) as f:
    data_in = json.loads('[' + ','.join(f.read().splitlines()) + ']')

pairs = []
for p in data_in:
    match p['gold_label']:
        case 'contradiction':
            label = 0
        case 'entailment':
            label = 1
        case _:
           continue
    if label == 1:
        x, y = p['sentence1'], p['sentence2']
        example = json.dumps({'query': x, 'pos': y})
        pairs.append(example)

with open(data_out, 'w') as f:
    f.write('\n'.join(pairs) + '\n')
