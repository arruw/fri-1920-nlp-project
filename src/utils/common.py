def read_tokens(inputFile):
  with open(inputFile) as f:
    return list(map(lambda l: tuple(l.replace('\n', '').split('\t')[:-1]), f.readlines()[7:]))

def group_by(iterator, key):
  from itertools import groupby
  return list(map(lambda g: (g[0], list(g[1])), groupby(sorted(iterator, key=key), key=key)))