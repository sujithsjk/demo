# -*- coding: utf-8 -*-


import pandas as pd
#from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import pickle
col_names = ['R', 'G', 'B', 'Class']
# load dataset
pima = pd.read_csv("D:\Machine learning\heroku_RF\RF.csv", header=1, names=col_names)
feature_cols = ['R', 'G', 'B']
label=['Class']
X = pima[feature_cols] # Features
y = pima[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, random_state=0)
clf = clf.fit(X_train,y_train)
pickle.dump(clf, open('model.pkl','wb'))
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))