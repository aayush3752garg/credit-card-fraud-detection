import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
from termcolor import colored as cl # text customization
import itertools # advanced tools

from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split # data split
from sklearn.tree import DecisionTreeClassifier # Decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.linear_model import LogisticRegression # Logistic regression algorithm
from sklearn.svm import SVC # SVM algorithm
from sklearn.ensemble import RandomForestClassifier # Random forest tree algorithm


from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import f1_score # evaluation metric
import streamlit as st
df = pd.read_csv('creditcard.csv')
df.drop('Time', axis = 1, inplace = True)
# print(df.shape)
# print(df.describe())

# print(df.head())
# df.hist(figsize = (20, 20))
# plt.show()
# print(df.tail())
cases = len(df)
nonfraud_count = len(df[df.Class == 0])
fraud_count = len(df[df.Class == 1])
fraud_percentage = round(fraud_count/nonfraud_count*100, 2)

# print(cl('CASE COUNT', attrs = ['bold']))
# print(cl('--------------------------------------------', attrs = ['bold']))
# print(cl('Total number of cases are {}'.format(cases), attrs = ['bold']))
# print(cl('Number of Non-fraud cases are {}'.format(nonfraud_count), attrs = ['bold']))
# print(cl('Number of Non-fraud cases are {}'.format(fraud_count), attrs = ['bold']))
# print(cl('Percentage of fraud cases is {}'.format(fraud_percentage), attrs = ['bold']))
# print(cl('--------------------------------------------', attrs = ['bold']))

sc = StandardScaler()
amount = df['Amount'].values

df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

# print(cl(df['Amount'].head(10), attrs = ['bold']))
X = df.drop('Class', axis = 1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# print(cl('X_train samples : ', attrs = ['bold']), X_train[:1])
# print(cl('X_test samples : ', attrs = ['bold']), X_test[0:1])
# print(cl('y_train samples : ', attrs = ['bold']), y_train[0:20])
# print(cl('y_test samples : ', attrs = ['bold']), y_test[0:20])
# 1. Decision Tree

#tree_model = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
#tree_model.fit(X_train, y_train)
# tree_yhat = tree_model.predict(X_test)

# 2. K-Nearest Neighbors

n = 5

knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)
# knn_yhat = knn.predict(X_test)

# 3. Logistic Regression

#lr = LogisticRegression()
#lr.fit(X_train, y_train)
# lr_yhat = lr.predict(X_test)

# 4. SVM 

#svm = SVC()
#svm.fit(X_train, y_train)
# svm_yhat = svm.predict(X_test)

# 5. Random Forest Tree

#rf = RandomForestClassifier(max_depth = 4)
#rf.fit(X_train, y_train)
# rf_yhat = rf.predict(X_test)
# print(cl('ACCURACY SCORE', attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, tree_yhat)), attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('Accuracy score of the KNN model is {}'.format(accuracy_score(y_test, knn_yhat)), attrs = ['bold'], color = 'green'))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)), attrs = ['bold'], color = 'red'))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('Accuracy score of the SVM model is {}'.format(accuracy_score(y_test, svm_yhat)), attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('Accuracy score of the Random Forest Tree model is {}'.format(accuracy_score(y_test, rf_yhat)), attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('F1 SCORE', attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, tree_yhat)), attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('F1 score of the KNN model is {}'.format(f1_score(y_test, knn_yhat)), attrs = ['bold'], color = 'green'))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)), attrs = ['bold'], color = 'red'))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('F1 score of the SVM model is {}'.format(f1_score(y_test, svm_yhat)), attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))
# print(cl('F1 score of the Random Forest Tree model is {}'.format(f1_score(y_test, rf_yhat)), attrs = ['bold']))
# print(cl('------------------------------------------------------------------------', attrs = ['bold']))


# def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
#     title = 'Confusion Matrix of {}'.format(title)
#     if normalize:
#         cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

#     plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation = 45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment = 'center',
#                  color = 'white' if cm[i, j] > thresh else 'black')

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# Compute confusion matrix for the models

# tree_matrix = confusion_matrix(y_test, tree_yhat, labels = [0, 1]) # Decision Tree
# knn_matrix = confusion_matrix(y_test, knn_yhat, labels = [0, 1]) # K-Nearest Neighbors
# lr_matrix = confusion_matrix(y_test, lr_yhat, labels = [0, 1]) # Logistic Regression
# svm_matrix = confusion_matrix(y_test, svm_yhat, labels = [0, 1]) # Support Vector Machine
# rf_matrix = confusion_matrix(y_test, rf_yhat, labels = [0, 1]) # Random Forest Tree

# Plot the confusion matrix

# plt.rcParams['figure.figsize'] = (6, 6)

# # 1. Decision tree

# tree_cm_plot = plot_confusion_matrix(tree_matrix, 
#                                 classes = ['Non-Default(0)','Default(1)'], 
#                                 normalize = False, title = 'Decision Tree')
# plt.savefig('tree_cm_plot.png')
# plt.show()

# 2. K-Nearest Neighbors

# knn_cm_plot = plot_confusion_matrix(knn_matrix, 
#                                 classes = ['Non-Default(0)','Default(1)'], 
#                                 normalize = False, title = 'KNN')
# plt.savefig('knn_cm_plot.png')
# plt.show()

# 3. Logistic regression

# lr_cm_plot = plot_confusion_matrix(lr_matrix, 
#                                 classes = ['Non-Default(0)','Default(1)'], 
#                                 normalize = False, title = 'Logistic Regression')
# plt.savefig('lr_cm_plot.png')
# plt.show()

# # 4. Support Vector Machine

# svm_cm_plot = plot_confusion_matrix(svm_matrix, 
#                                 classes = ['Non-Default(0)','Default(1)'], 
#                                 normalize = False, title = 'SVM')
# plt.savefig('svm_cm_plot.png')
# plt.show()

# 5. Random forest tree

# rf_cm_plot = plot_confusion_matrix(rf_matrix, 
#                                 classes = ['Non-Default(0)','Default(1)'], 
#                                 normalize = False, title = 'Random Forest Tree')
# plt.savefig('rf_cm_plot.png')
# plt.show()


st.title("Credit Card Fraud Detection Model")

input_df = st.text_input("Enter ALL THE Values")
input_df_splited= input_df.split(",")
submit= st.button("Submit")

if submit:
    features = np.asarray(input_df_splited, dtype=np. float64) 
    prediction = knn.predict(features.reshape(1,-1))
    if prediction[0] == 0:
        st.write("Non Fraud Transaction")
    else:
        st.write("Fraud Transaction")
