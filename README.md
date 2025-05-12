# 4361-Final-Project-Team-9
Team 6 final project for Spring 2025 CS4361 Machine Learning 

KNN_EMNIST_Classifier.py written by - David S. Rivera-Rivera

dtree.py, logisticRegression.py, & svm.py - Isaac Hita

======================= INSTRUCTIONS =======================

Instructions on how to run the KNN_EMNIST_Classifier.py code:
    1) Install the emnist-balanced-mapping.txt and the zip containing emnist-balanced-test.csv and emnist-balanced-train.csv
    2) Install KNN_EMNIST_Classifier.py to the same directory as the dataset files
    3) Execute the program

After step 1 for Install all the python files and download 

https://www.kaggle.com/datasets/nikbearbrown/tmnist-alphabet-94-characters/data (necessary for tmnist)

https://www.kaggle.com/datasets/crawford/emnist/data?select=emnist-byclass-test.csv  (Optional already in the github zip)

Also go to {USAGE} for the SVM, LR, DT files on how to run


======================= PROGRAM DETAILS =======================

During the execution of KNN_EMNIST_Classifier.py the following will be done:
    i) the program will output status to the console as well as other performance metrics throughout its execution
    ii) the program will display 2 graphs, the confusion matrix, and bar plot showing the # of misclassifications per class
    iii) the program will out a performance report using sklearns classification report
    iv) a dictionary will be outputted containing the decoded labels and associated characters to better understand the classification report
    v) lastly, the program will output any SINGLE misclassification per class that contributes to more than 25% of the total misclassification for one label we will output that result. For example, if 'p' is classified as '9' more than or equal to 25% of 400 samples for 'p', then the program will print that result


1. dtree.py (Decision Tree Classifier)
    i) The program will output status to the console and display performance metrics (e.g., accuracy, precision, recall) throughout the execution.

    ii) The program will display two graphs:
        - The confusion matrix to show misclassifications between predicted and actual labels.
        - A Decision Tree plot visualizing the structure of the decision tree (zoomed into the first 3 levels for clarity).

    iii) The program will output a performance report using sklearn’s classification_report, showing metrics such as precision, recall, and F1-scores for each class.



2. logisticRegression.py (Logistic Regression Classifier)
    i) The program will output status updates to the console and show performance metrics as the model trains and evaluates.

    ii) The program will display two graphs:
        - The confusion matrix to highlight how well the model predicted each class.
        - A Scree plot showing the explained variance ratio for each principal component in PCA.

    iii) The program will output a performance report using sklearn’s classification_report, detailing precision, recall, and F1-scores.


3. svm.py (Support Vector Machine Classifier)
    i) The program will output status to the console and provide performance metrics throughout its execution.

    ii) The program will display three graphs:
        - The confusion matrix to visualize the model’s predictions vs actual values.
        - An F1-score bar chart showing the performance for each class.
        - A Scree plot illustrating the explained variance from PCA.

    iii) The program will output a performance report using sklearn’s classification_report, providing a detailed evaluation.


======================= USAGE (For running SVM, LR, DT) =======================

For easier usage, you can execute any of the three scripts by using the following commands:

[--quick] is a test run, to see if the code can run for each model, either ignore or don't copy

1. Decision Tree (dtree.py)
To run the Decision Tree Classifier:

python dtree.py --mode emnist --quick

python dtree.py --mode emnist

python dtree.py --mode tmnist_symbols

3. Logistic Regression (logisticRegression.py)
To run the Logistic Regression Classifier:

python logisticRegression.py --mode emnist --quick

python logisticRegression.py --mode emnist

python logisticRegression.py --mode tmnist_symbols

5. Support Vector Machine (svm.py)
To run the Support Vector Machine Classifier:

python svm.py --mode emnist --quick

python svm.py --mode emnist

python svm.py --mode tmnist_symbols

Arguments for each script:
--mode: Choose between emnist (digits and letters) or tmnist_symbols (symbols only).

--quick: If specified, the program will use a smaller dataset and fewer PCA components for quicker execution.

--emnist_train_csv, --emnist_test_csv, --mapping_txt, --tmnist_csv: Specify the paths to the respective data files.

--model_dir: Directory to save or load the trained models.


======================= SOURCES =======================

Sources used for KNN_EMNIST_Classifier.py:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits_active_learning.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-digits-active-learning-py
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-digits-py
https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_agglomeration.html#sphx-glr-auto-examples-cluster-plot-digits-agglomeration-py
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html
https://stackoverflow.com/questions/25551977/retrieve-misclassified-documents-using-scikitlearn
https://stackoverflow.com/questions/32461246/how-to-get-top-3-or-top-n-predictions-using-sklearns-sgdclassifier
https://stackoverflow.com/questions/61526287/how-to-add-correct-labels-for-seaborn-confusion-matrix
https://seaborn.pydata.org/generated/seaborn.heatmap.html
https://www.kaggle.com/code/kadriyeaksakal/confusion-matrix-with-knn-algorithm
https://arxiv.org/abs/1702.05373

Sources used for logisticRegression.py:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
https://seaborn.pydata.org/generated/seaborn.heatmap.html
https://stackoverflow.com/questions/36988123/how-to-interpret-pca-scree-plot
https://stackoverflow.com/questions/28931224/plotting-heatmap-of-classification-report-in-python
https://stackoverflow.com/questions/11874767/plotting-a-roc-curve-in-scikit-learn

Sources used for svm.py:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
https://seaborn.pydata.org/generated/seaborn.barplot.html
https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html
https://stackoverflow.com/questions/39465687/how-to-plot-decision-boundary-for-svm
https://stackoverflow.com/questions/42010721/how-to-plot-a-decision-boundary-for-multiclass-classification
https://stackoverflow.com/questions/25551977/retrieve-misclassified-documents-using-scikitlearn

Sources used for dtree.py:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
https://seaborn.pydata.org/generated/seaborn.heatmap.html
https://stackoverflow.com/questions/28931224/plotting-heatmap-of-classification-report-in-python
https://stackoverflow.com/questions/61526287/how-to-add-correct-labels-for-seaborn-confusion-matrix
