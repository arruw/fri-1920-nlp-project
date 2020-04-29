import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE as sampler
# from imblearn.under_sampling import ClusterCentroids as sampler

def main():
  df = pd.read_csv('cache/baseline.csv')
  df = df.replace({'type': {'PER': 1, 'ORG': 2, 'LOC': 3}})

  CA_majority = df[['actual', 'id']].groupby(['actual']).agg(['count'])['id']['count'][0]/len(df)

  X_train, X_test, y_train, y_test = train_test_split(df[['type', 'len', 'min', 'max', 'avg', 'var']], df[['actual']], test_size=0.15)
  # X_train, y_train = sampler().fit_resample(X_train, y_train)

  clf = SVC()
  clf.fit(X_train, y_train)

  CA_baseline = accuracy_score(y_test, clf.predict(X_test))

  print(f'Majority CA: {CA_majority}')
  print(f'Baseline CA: {CA_baseline}')

if __name__ == "__main__":
  main()