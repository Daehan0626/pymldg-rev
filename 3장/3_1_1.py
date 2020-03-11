from sklearn.base import BaseEstimator
import numpy as np

class MyDummyClassifier(BaseEstimator):
    def fit(self,x,y=None):
        pass
    def predict(self,x):
        pred = np.zeros((x.shape[0],1))
        # pred 는 다 0
        for i in range(x.shape[0]):
            if x['Sex'].iloc[i] ==1 :
                pred[i] = 0
            else :
                pred[i] =1
        return pred

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행. 
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


titanic_df = pd.read_csv('/home/daehan/Workspace/ML/pymldg-rev/3장/titanic_train.csv')
print(titanic_df.head(3))

y_titanic_df = titanic_df['Survived']
x_titanic_df = titanic_df.drop('Survived',axis=1)
x_titanic_df = transform_features(x_titanic_df)
x_train, x_test, y_train, y_test = train_test_split(x_titanic_df,y_titanic_df,test_size=0.2,random_state=0)

myclf = MyDummyClassifier()
myclf.fit(x_train,y_train)

mypredictions = myclf.predict(x_test)
print("정확도:",accuracy_score(y_test,mypredictions))