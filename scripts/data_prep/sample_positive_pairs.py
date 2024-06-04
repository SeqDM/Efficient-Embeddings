import sys
from random import randrange, random

def random_crop(text, l_min=10, l_max=100):
    l_text = len(text)
    l_max = min(l_max, l_text)
    l_min = min(l_min, l_max)
    l_crop = randrange(l_min, l_max + 1)
    start = randrange(0, l_text - l_crop + 1)
    stop = start + l_crop
    return text[start:stop]

def random_drop(text, drop_prob = 0.1):
    drop_text = [t for t in text if drop_prob < random()]
    return drop_text

if __name__ == '__main__':
    text = sys.argv[1]
    n_pairs = int(sys.argv[2])
    with open(text) as f:
        text = f.read()

    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    text = text.split(' ')

    for _ in range(n_pairs):
        x = ' '.join(random_drop(random_crop(text)))
        y = ' '.join(random_drop(random_crop(text)))
        print()
        print(f'{x}   #   {y}')
