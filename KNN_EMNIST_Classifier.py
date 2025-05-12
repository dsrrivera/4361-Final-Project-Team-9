#CS4361 Machine Learning Final Project - Code written by: David S. Rivera-Rivera

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from scipy import stats
import sklearn.metrics.pairwise as kernel_lib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import time

def load_dataset(filename):
  return pd.read_csv(filename, header=None)

#load training data
train_csv = load_dataset('emnist-balanced-train.csv')
X_train = train_csv.iloc[:,1:] / 255.0 
y_train = train_csv.iloc[:,0]
print('Done loading training samples.\n')

#load testing data
test_csv = load_dataset('emnist-balanced-test.csv')
X_test = test_csv.iloc[:,1:] / 255.0
y_test = test_csv.iloc[:,0]
print('Done loading testing samples.\n')

emnist_knn = KNeighborsClassifier(n_neighbors=5)
emnist_knn.fit(X_train, y_train)
print('Done fitting KNN.\n')

print("Training size:", len(y_train))
print("Testing size:", len(y_test), "\n")

#the class names come in 0-46 and are encoded as their respective unicode values, this makes a dict that maps 0-46 to the characters
decoded_class_names = dict()
with open('emnist-balanced-mapping.txt', 'r') as file:
    for line in file:
        curr_word = line.split()
        decoded_class_names[curr_word[0]] = chr(int(curr_word[1]))

print("Decoded class label to corresponding letter:")
print(decoded_class_names, '\n')

#set up x_test as numpy for optimization and predict on the testing set
X_np_array = X_test.values
y_predicted = emnist_knn.predict(X_np_array)


print("Classification Report for EMNIST KNN:")
print(classification_report(y_test, y_predicted))

#get confusion matrix, this is used for the majority of the plots/analysis later in the code
confusion_matrix = confusion_matrix(y_test, y_predicted, labels = emnist_knn.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=decoded_class_names.values())
plt.figure(figsize=(18, 10))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=decoded_class_names.values(), yticklabels=decoded_class_names.values())
plt.title("Confusion Matrix for KNN on EMNIST for k=5")
print("Currently displaying confusion matrix plot.\n")
plt.show()

#this code iterates through the 2d confusion matrix, sums the rows, identifies the diagonal (correct # predictions), and gets the # of misclassifications
misclassifications = []
largest_misclassified_classes = [(0,0) for i in range(len(confusion_matrix))]
sumsa = []
for row in range(len(confusion_matrix)):
  current_sum = 0
  diagonal = 0

  for column in range(len(confusion_matrix[0])):
    current_sum += confusion_matrix[row, column]
    if row == column: 
      diagonal = confusion_matrix[row, column]
    if row != column and confusion_matrix[row, column] > largest_misclassified_classes[row][1]: 
      largest_misclassified_classes[row] = (decoded_class_names.get(str(column)), confusion_matrix[row, column])

  misclassifications.append(current_sum - diagonal)
  sumsa.append(current_sum)

#display the misclassification bar plot
plt.figure(figsize=(10, 4))
plt.bar(range(len(misclassifications)), misclassifications)
plt.xticks(range(len(misclassifications)), decoded_class_names.values())
plt.xlabel("Letter and Digit Classes")
plt.ylabel("Number of Misclassifications")
plt.title("# of Misclassifications By Class")
print("Currently displaying misclassification plot.\n")
plt.show()

#if any misclassification contributes to more than 25% of the total misclassification for one label we will output that result
#so, if p is classified as 9 more than or equal to 25%, then we would print that result
# print(largest_misclassified_classes)
for i in range(len(largest_misclassified_classes)):
  if (largest_misclassified_classes[i][1] / 400) >= .25:
    print(f"Class {decoded_class_names.get(str(i))} classified as {largest_misclassified_classes[i][0]}, {largest_misclassified_classes[i][1]} times out of 400.")

print("\nProgram finished executing.")