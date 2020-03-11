from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 

class MyFakeClassifier(BaseEstimator):
    def fit(self, x, y):
        pass

    def predict(self,x):
        return np.zeros((len(x),1),dtype=bool)

digit = load_digits()
#print(digit)
y = (digit.target == 7 ).astype(int)
x_train, x_test, y_train, y_test = train_test_split(digit.data, y, random_state=11)

print(y_test.shape)
print(pd.Series(y_test).value_counts())

fakeclf = MyFakeClassifier()
fakeclf.fit(x_train,y_train)
fakerpred = fakeclf.predict(x_test)
print("fake정확도:",accuracy_score(y_test,fakerpred))
