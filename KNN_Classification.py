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

## Setup arrays to store train and test accuracies
neighbors = np.arange(2, 5)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#identifying k=3 as an optimum solution
#run k-nearest neighbor algorithm with k=3
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)
print("Train_Score: {}".format(knn.score(X_train, y_train)))
print("Test_Score: {}".format(knn.score(X_test, y_test)))

#observing the decision boundaries for knn
from mlxtend.plotting import plot_decision_regions

X_b = X_train[100:150,[2,5]]
y_b = y_train[100:150]

knn = KNeighborsClassifier(10)
knn.fit(X_b, y_b) 

plot_decision_regions(X_b, y_b, clf = knn)
