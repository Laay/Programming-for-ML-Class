# Python-Machine-Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data pre-processing

file1 = pd.read_csv("C:/Users/laayt/Downloads/audit_risk.csv")
file2 = pd.read_csv("C:/Users/laayt/Downloads/trial.csv")

#merging files on 'Sector_score' variable
data  =  pd.merge(file1, file2, on = 'Sector_score')

#removing NA
data = data.dropna()

#create a dataframe having highly correlated features
corr = data.corr().abs()
corr = corr.unstack()
corr = corr.sort_values(kind="quicksort")
corr[(corr > 0.9) & (corr <1)]

# deleted redundent variables
new_data = data.drop(['Score_x', 'Score_y', 'numbers_y', 'numbers_x','District_Loss', 'Risk_C', 'Risk_F', 'Risk_B', 'PARA_B_x',
'PARA_B_y', 'PARA_A_x', 'LOSS_SCORE'], axis =1)

#setting target variable and feature variables
y = data['Risk_x']
X = data.drop('Risk_x',axis =1)

#applying minmax scaling and train-test split

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#deviding train test split
X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)

#applying Linear Support Vector machine algorithm

from sklearn.svm import SVC, LinearSVC
C_space = [0.1, 1, 10]
param_svm = {'C' : C_space}
linear_svm = LinearSVC()

linear_svm = GridSearchCV(linear_svm,param_svm)
linear_svm.fit(X_train, y_train)

# Print the tuned parameters and score

print("Tuned Linear SVM Parameters: {}".format(linear_svm.best_params_))
print("Best score is {}".format(linear_svm.best_score_))

linear_svm = LinearSVC(C=10)
linear_svm.fit(X_train, y_train)

print("Train_Score: {}".format(logreg.#score(X_train, y_train)))
print("Test_Score: {}".format(logreg.score(X_test, y_test)))

#confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = linear_svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

y_pred_train_linear_svm = linear_svm.predict(X_train)
y_pred_test_linear_svm = linear_svm.predict(X_test)

#precision score
print("p_score_train: {}".format(precision_score(y_train, y_pred_train_linear_svm)))
print("p_score_test: {}".format(precision_score(y_test, y_pred_test_linear_svm)))

#plotting decision boundaries
%matplotlib inline

from mlxtend.plotting import plot_decision_regions

X_b = X_train[1:20000,[1,3]]
y_b = y_train[1:20000]
linear_svm = LinearSVC()
linear_svm.fit(X_b,y_b)

plot_decision_regions(X_b, y_b, linear_svm)
