import os
from operator import itemgetter
from pprint import pprint
from utils.common import read_tokens, group_by

def extract_labeled_nouns(tokens):
  nouns = group_by(filter(lambda t: t[3] != '_', tokens), key=itemgetter(6))
  sentiments = []
  for _, noun in nouns:
    sentiment = list(filter(lambda n: n[4] != '_', noun))
    if not sentiment: continue
    sentiment = sentiment[-1]
    sentiments.append((sentiment[3][0:3], int(sentiment[4].split(' - ')[0])))
  return sentiments
  
nouns = []
for file in os.listdir('./dataset'):
  tokens = read_tokens(os.path.join('./dataset', file))
  nouns += extract_labeled_nouns(tokens)

# Numbers are a bit smaller then what is reported in the report.
# extract_labeled_nouns should be fixed.
print("=== Fine grained ===")
stats_fine = [(x[0], len(x[1])) for x in group_by(nouns, key=itemgetter(0, 1))]
pprint(stats_fine)

print("=== Noun type ===")
stats_noun = [(x[0], len(x[1])) for x in group_by(nouns, key=itemgetter(0))]
pprint(stats_noun)

print("=== Sentiment level ===")
stats_sentiment = [(x[0], len(x[1])) for x in group_by(nouns, key=itemgetter(1))]
pprint(stats_sentiment)

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.DataFrame(nouns, columns=["Type", "Sentiment level"])
g = sns.FacetGrid(df, row="Type", row_order=["LOC", "ORG", "PER"], sharex=True, sharey=True)
g = g.map(plt.hist, "Sentiment level", bins=[1,2,3,4,5,6], edgecolor="w", align="mid", log=True, orientation="horizontal")
plt.savefig('./resources/sentiment_distributions.png')
plt.show()