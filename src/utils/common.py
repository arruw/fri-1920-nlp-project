from operator import itemgetter
from typing import List, Tuple, Dict, Set
from nltk.stem import PorterStemmer

class Entity:
  def __init__(self, entityId: int):
    self.id: int = entityId
    self.freq: int = 0
    self.type: str = None
    self.substrings: List[Tuple[int, int]] = []
    self.sentiment: int = None
    self._nameParts: Set[str] = []
    self._coref: str = None

  def _addNamePart(self, part: str, coref):
    if self._coref == None or self._coref == coref:
      self._coref = coref
      self._nameParts.append(part)

  def getName(self) -> str:
    return ' '.join(self._nameParts)

  def update(self, token: Tuple):
    self.freq += 1
    if token[3] != '_':
      self.type = token[3][0:3]
      self._addNamePart(token[2], token[5])
    self.substrings.append(tuple(map(int, token[1].split('-'))))
    if token[4] != '_':
      self.sentiment = int(token[4].split(' - ')[0])

def read_tokens(inputFile):
  with open(inputFile) as f:
    return list(map(lambda l: tuple(l.replace('\n', '').split('\t')[:-1]), f.readlines()[7:]))

def group_by(iterator, key):
  from itertools import groupby
  return list(map(lambda g: (g[0], list(g[1])), groupby(sorted(iterator, key=key), key=key)))

def extract_entities(tokens: List[Tuple]) -> List[Entity]:
  """
  Extract entities
  """
  results = dict()
  for token in tokens:
    # print(f'\t\t{token[0]}')
    if token[6] == '_': continue
    entityIds = map(lambda x: int(x[2:-1]), token[6].split('|'))
    for entityId in entityIds:
      entity: Entity = results.get(entityId, Entity(entityId))
      entity.update(token)
      results[entityId] = entity
  return list(results.values())