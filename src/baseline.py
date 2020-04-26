import os
import pandas as pd
from typing import List
from polyglot.text import Text, Sentence
from utils.common import extract_entities, read_tokens, Entity

def extract_text(inputFilePath: str) -> Text:
  lines: List[str]
  with open(inputFilePath) as f:
    lines = list(map(lambda l: l.replace('\n', ''), f.readlines()))
  for line in lines:
    if not line.startswith('#Text='): continue
    return Text(line[6:])

def extract_entity_context(entity: Entity, text: Text) -> str:
  def is_sentence_in_context(x: Sentence) -> bool:
    return len(list(filter(lambda ss: x.start <= ss[0] and ss[1] <= x.end, entity.substrings))) > 0
  return ' '.join(map(str, filter(is_sentence_in_context, text.sentences)))

data = []
for file in os.listdir('./dataset'):
  docId = int(file[:-4])
  docPath = os.path.join('./dataset', f'{docId}.tsv')
  print(f'Extracting text... {{docId={docId}}}')
  text = extract_text(docPath)
  # print('Extracting entities...')
  entities = extract_entities(read_tokens(docPath))
  for entity in entities:
    if not entity.sentiment: continue
    # print(f'\tExtracting entity context... {{entityId={entity.entityId}}}')
    data.append((
      f'{docId}-{entity.id}',
      entity.getName(),
      entity.type,
      extract_entity_context(entity, text),
      entity.sentiment
    ))
df = pd.DataFrame(data, columns=['id', 'name', 'type', 'context', 'sentiment'])
df.to_csv(path_or_buf='cache/baseline.csv', index=False)

df.describe()