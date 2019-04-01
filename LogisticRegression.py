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

#calling Logistic Regression model
from sklearn.linear_model import LogisticRegression
c_range = [0.001, 0.01, 0.1, 1, 10]
param_grid = {'C' : c_range}

logreg = LogisticRegression()

logreg = GridSearchCV(logreg,param_grid)
logreg.fit(X_train, y_train)

# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(logreg.best_params_))
print("Best score is {}".format(logreg.best_score_))

##taking C=10
logreg = LogisticRegression(C=10)

logreg.fit(X_train, y_train)
print("Train_Score: {}".format(logreg.score(X_train, y_train)))
print("Test_Score: {}".format(logreg.score(X_test, y_test)))

#train_accuracy vs test-accuracy chart
c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_score_log = []
test_score_log = []

for c in c_range:
    logreg = LogisticRegression(C = c)
    logreg.fit(X_train, y_train)
    train_score_log.append(logreg.score(X_train,y_train))
    test_score_log.append(logreg.score(X_test,y_test))
    
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(c_range, train_score_log, label = 'Train score')
plt.plot(c_range, test_score_log, label = 'Test score')
plt.legend()
plt.xlabel('Regularization parameter: C')
plt.ylabel('Accuracy')
plt.xscale('log')

#decision boundaries for LogisticRegression
%matplotlib inline

from mlxtend.plotting import plot_decision_regions

X_b = X_train[10:200, [1,3]]
y_b = y_train[10:200]

lreg = LogisticRegression()
lreg.fit(X_b, y_b) 

plot_decision_regions(X_b, y_b, clf = lreg)
