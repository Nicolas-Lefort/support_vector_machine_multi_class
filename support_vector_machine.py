# data from https://archive.ics.uci.edu/ml/datasets/smartphone-based+recognition+of+human+activities+and+postural+transitions
# Smartphone measurements vs human activities
# SVM using gaussian kernel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from utils import encode, plot_correlation

# import data
df = pd.read_csv('smartphones_data.csv')
# define target feature
target = "Activity"
# prepare the data
df, df_numeric, _, _, _ = encode(df=df.copy(), ordinal_features=None, target=target)
# plot correlations
plot_correlation(df_numeric.join(df[target]), target, 10)
# define x and y
X, y = df.drop(columns=target), df[target]
# split train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# define the model
SVC_Gaussian = SVC(kernel='rbf', gamma='auto')
# fit to train data
SVC_Gaussian = SVC_Gaussian.fit(X_train, y_train)
# predict test data
y_pred = SVC_Gaussian.predict(X_test)
# Precision, recall, f-score from the multi-class support function
print(classification_report(y_test, y_pred))
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
disp.plot(cmap=plt.cm.Blues)
plt.show()