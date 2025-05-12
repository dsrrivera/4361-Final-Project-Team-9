import os
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def load_emnist_train_test(train_csv, test_csv, mapping_txt, quick=False):
    """
    Load EMNIST ByClass train/test CSVs and mapping file.
    Returns X_train, y_train, X_test, y_test.
    """
    df_train = pd.read_csv(train_csv, header=None)
    df_test = pd.read_csv(test_csv, header=None)
    mapping = {}
    with open(mapping_txt, 'r') as f:
        for line in f:
            idx, code = map(int, line.strip().split())
            mapping[idx] = chr(code)
    X_train = df_train.iloc[:, 1:].values
    y_train = np.array([mapping[l] for l in df_train.iloc[:, 0].values])
    X_test = df_test.iloc[:, 1:].values
    y_test = np.array([mapping[l] for l in df_test.iloc[:, 0].values])
    if quick:
        keep_idx = []
        for c in np.unique(y_train):
            idxs = np.where(y_train == c)[0]
            keep_idx.extend(idxs[:200])
        X_train = X_train[keep_idx]
        y_train = y_train[keep_idx]
    return X_train, y_train, X_test, y_test


def load_tmnist_symbols(tmnist_csv):
    """
    Load TMNIST CSV, filter to symbol classes only.
    Returns X_train, y_train, X_test, y_test using 80/20 split.
    """
    df = pd.read_csv(tmnist_csv)
    df_sym = df[~df['labels'].str.isalnum()]
    X = df_sym.iloc[:, 2:].values
    y = df_sym['labels'].values
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def main():
    parser = argparse.ArgumentParser(
        description="Train/test Decision-Tree on EMNIST or TMNIST symbols with hyperparameter tuning and visualization"
    )
    parser.add_argument('--mode', choices=['emnist','tmnist_symbols'], default='emnist',
                        help="'emnist' for digits+letters, 'tmnist_symbols' for symbols only")
    parser.add_argument('--quick', action='store_true', help='Use small subset for EMNIST')
    parser.add_argument('--emnist_train_csv', default='emnist-balanced-train.csv')
    parser.add_argument('--emnist_test_csv', default='emnist-balanced-test.csv')
    parser.add_argument('--mapping_txt', default='emnist-balanced-mapping.txt')
    parser.add_argument('--tmnist_csv', default='94_character_TMNIST.csv')
    parser.add_argument('--model_dir', default='.')
    args = parser.parse_args()

    # Load data
    if args.mode == 'emnist':
        X_train, y_train, X_test, y_test = load_emnist_train_test(
            args.emnist_train_csv, args.emnist_test_csv, args.mapping_txt, quick=args.quick
        )
        model_name = 'emnist_dt_pipeline.joblib'
    else:
        X_train, y_train, X_test, y_test = load_tmnist_symbols(args.tmnist_csv)
        model_name = 'tmnist_symbols_dt_pipeline.joblib'

    # PCA components
    n_components = 10 if args.quick else 100
    MODEL_PATH = os.path.join(args.model_dir, model_name)

    # Define base pipeline
    pipeline_base = Pipeline([
        ('scaler', MinMaxScaler()),
        ('pca', PCA(n_components=n_components)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ])

    # Hyperparameter distributions
    param_dist = {
        'dt__max_depth': [None, 10, 20, 40],
        'dt__min_samples_split': [2, 5, 10],
        'dt__min_samples_leaf': [1, 2, 4],
        'dt__max_features': [None, 'sqrt']
    }

    # Load or tune model
    if os.path.exists(MODEL_PATH) and not args.quick:
        pipeline = joblib.load(MODEL_PATH)
        print(f"Loaded Decision-Tree model for {args.mode}.")
    else:
        cv = 2 if args.quick else 3
        n_iter = 5 if args.quick else 30
        search = RandomizedSearchCV(
            pipeline_base,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        print(f"Tuning Decision-Tree ({args.mode}) with {n_iter} iterations...\n")
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        print(f"Best parameters: {search.best_params_}\n")
        if not args.quick:
            joblib.dump(pipeline, MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}.")

    # Visualize tree (zoomed-in view, first 3 levels)
    classes = sorted(np.unique(y_train), key=lambda c: ord(c))
    feature_names = [f"PC{i+1}" for i in range(n_components)]
    fig, ax = plt.subplots(figsize=(40, 20), dpi=40)
    plot_tree(
        pipeline.named_steps['dt'],
        max_depth=3,
        feature_names=feature_names,
        class_names=classes,
        filled=True,
        rounded=True,
        fontsize=12,
        ax=ax
    )
    plt.title("Decision Tree Visualization (depth=3)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    classes_cm = sorted(np.unique(y_test), key=lambda c: ord(c))
    cm = confusion_matrix(y_test, y_pred, labels=classes_cm)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, fmt='d', cmap='Blues', xticklabels=classes_cm, yticklabels=classes_cm)
    plt.title(f"Decision-Tree Confusion Matrix ({args.mode})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
