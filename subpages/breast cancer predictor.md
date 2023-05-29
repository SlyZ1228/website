# Breast Cancer Predictor

This is a simple application of the K Nearest Neighbors classification algorithm using the UCI ML Breast Cancer Dataset (which can be found here: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer).


```python
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
```


```python
df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace = True)
df.drop(columns = 'id', inplace = True)

X = np.array(df.drop(columns = 'class'))
y = np.array(df['class'])
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)
```

    0.9857142857142858
    
