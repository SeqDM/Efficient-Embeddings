#!/bin/env python3

# Returns a jsonl file with lines of form:
#
# {'x': query, 'y': passage/answer}
#
# For example:
#
# {
#   "x": "does the word example get capitalized",
#   "y": "World should be capitalized when: 1  It is the first word in a
#         sentence; for example, 'World' should be capitalized when it is the first
#         word in a sentence.. 2  It is part of a proper name; for example, World War
#         II, World of Warcraft..",
# }

import sys
import json

data_in = sys.argv[1]
data_out = sys.argv[2]

with open(data_in) as f:
    data_in = json.load(f)

ids = set(data_in['query'])

examples = []

for i in ids:
    query = data_in['query'][i]
    passages = [p['passage_text'] for p in data_in['passages'][i] if \
                p['is_selected'] == 1]
    passage = passages[0] if passages else ''
    answer = data_in['answers'][i][0] if data_in['answers'][i] else ''
    if answer == 'No Answer Present.':
        answer = ''
    if answer:
        example = json.dumps({'query': query, 'pos': answer})
        examples.append(example)
    if passage:
        example = json.dumps({'query': query, 'pos': passage})
        examples.append(example)

with open(data_out, 'w') as f:
    f.write('\n'.join(examples) + '\n')
