#!/bin/env python3

import sys
import json
import csv

data_in = sys.argv[1]
data_out = sys.argv[2]

with open(data_in, 'r') as f:
    data_in = csv.reader(f, delimiter=',', quotechar='"')
    next(data_in, None)
    pairs = []
    for p in data_in:
        x, y, neg = p[0], p[1], p[2]
        example = json.dumps({'query': x, 'pos': y, 'neg': neg})
        pairs.append(example)

with open(data_out, 'w') as f:
    f.write('\n'.join(pairs) + '\n')
