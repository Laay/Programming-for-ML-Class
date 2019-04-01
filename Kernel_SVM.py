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

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)

%matplotlib inline

from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC, LinearSVC
import matplotlib.gridspec as gridspec
import itertools
C=1
clf1 = SVC(kernel='rbf', gamma=0.1, C=C)
clf2 = SVC(kernel='rbf', gamma=1, C=C)
clf3 = SVC(kernel='rbf', gamma=10, C=C)

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

labels = ['gamma: 0,1','gamma: 1','gamma: 10']
X_b = X_train[1:20000,[1,3]]
y_b = y_train[1:20000]

for clf, lab, grd in zip([clf1, clf2, clf3],
                         labels,
                         itertools.product([0, 1],
                         repeat=2)):
    clf.fit(X_b, y_b)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_b, y=y_b,
                                clf=clf, legend=2)
    plt.title(lab)
 
 from sklearn.svm import SVC

gammas = [0.1, 1, 10]
param_rbf = {'gamma' : gammas}

sv_class = SVC(kernel = 'rbf', C=1)
sv_class = GridSearchCV(sv_class,param_rbf)
sv_class.fit(X_train, y_train)

# Print the tuned parameters and score

print("Tuned RBF SVM Parameters: {}".format(sv_class.best_params_))
print("Best score is {}".format(sv_class.best_score_))
