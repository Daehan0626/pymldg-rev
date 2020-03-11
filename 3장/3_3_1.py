## 정밀도(precision) = TP / (FN+TP)
# TP / 예측 건수 - 예측 성능 확인
## 재현율(recall) = TP / (FN+TP)
# TP / 모든 데이터 건수  - TPR , Sensitivity\]

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score

def get_clf_eval(y_test,pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    print("Confusion_Matrix:\n",confusion)
    print("accuracy :{0:2.4f} , precision:{1:0.4f}, recall:{2:0.4f}, f1_score:{3:0.4f}".format(accuracy,precision,recall,f1))

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np


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
print(titanic_df)
y_titanic_df = titanic_df['Survived']

# extract Survied columm 
x_titanic_df = titanic_df.drop('Survived',axis=1)
# Data preprocesing 
x_titanic_df = transform_features(x_titanic_df)

x_train, x_test, y_train, y_test = train_test_split(x_titanic_df,y_titanic_df,test_size=0.2,random_state=11)

lr_clf = LogisticRegression()
lr_clf.fit(x_train,y_train)
pred = lr_clf.predict(x_test)
get_clf_eval(y_test,pred)

### Precision - Recall Trade OFF

# predict_proba(y_test) -> 각클래스 별 예측 확률

pred_proba=lr_clf.predict_proba(x_test)
print("\npred_proba:")
print(pred_proba[:3])

pred_proba_result = np.concatenate([pred_proba,pred.reshape(-1,1)],axis=1)
print("\npred_proba & result:")
print(pred_proba_result[:3])

from sklearn.preprocessing  import Binarizer
# Binarizer 는 thresh 값 기준으로 0, 1 나눔

x=[ [1, -1,  2],
    [2,  0,  0],
    [0,1.1,1.2] ]
binarizer = Binarizer(threshold=1.1)
print("\nbinarizer_result:")
print(binarizer.fit_transform(x))

custom_threshold=0.4

pred_proba_1 = pred_proba[:,1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) 
custom_predict = binarizer.transform(pred_proba_1)

get_clf_eval(y_test, custom_predict)


## control threshold

threshold=[0.4,0.45,0.5,0.55,0.6]

def get_eval_by_threshold(y_test,pred_proba_c1,thresholds):
    for custom_threshold in threshold:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) 
        custom_predict = binarizer.transform(pred_proba_1)
        get_clf_eval(y_test, custom_predict)

get_eval_by_threshold(y_test,pred_proba[:,1].reshape(-1,1),threshold)

## precision-recall curvve

from sklearn.metrics import precision_recall_curve

pred_proba_class1 =lr_clf.predict_proba(x_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test,pred_proba_class1)
print("threshold shpape :",thresholds.shape)

thr_index = np.arange(0,thresholds.shape[0],15)
print("thr_index:",thr_index)
print("precision:",np.round(precision[thr_index],3))
print("precision:",np.round(recall[thr_index],3))

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

def precision_recall_curve_plot(y_test , pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    
precision_recall_curve_plot( y_test, lr_clf.predict_proba(x_test)[:, 1] )


