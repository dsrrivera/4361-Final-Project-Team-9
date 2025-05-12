import os
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns


def load_emnist_train_test(train_csv, test_csv, mapping_txt, quick=False):
    """
    Load EMNIST CSVs and mapping. If quick=True, sample a small balanced subset of the train set for fast testing.
    The test set remains unchanged.
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
    Load TMNIST CSV and filter to symbol classes only (non-alphanumeric).
    Returns X_train, y_train, X_test, y_test using a standard 80/20 split.
    """
    df = pd.read_csv(tmnist_csv)
    df_sym = df[~df['labels'].str.isalnum()]
    X = df_sym.iloc[:, 2:].values
    y = df_sym['labels'].values
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def main():
    parser = argparse.ArgumentParser(
        description="Train/test Logistic Regression on EMNIST or TMNIST symbols"
    )
    parser.add_argument('--mode', choices=['emnist', 'tmnist_symbols'], default='emnist',
                        help="'emnist' for digits+letters, 'tmnist_symbols' for symbols only")
    parser.add_argument('--quick', action='store_true',
                        help='Use a small subset of train and fewer PCA comps for EMNIST')
    parser.add_argument('--emnist_train_csv', default='emnist-balanced-train.csv',
                        help='EMNIST balanced training CSV path')
    parser.add_argument('--emnist_test_csv', default='emnist-balanced-test.csv',
                        help='EMNIST balanced test CSV path')
    parser.add_argument('--mapping_txt', default='emnist-balanced-mapping.txt',
                        help='EMNIST balanced mapping file path')
    parser.add_argument('--tmnist_csv', default='94_character_TMNIST.csv',
                        help='TMNIST CSV path')
    parser.add_argument('--model_dir', default='.',
                        help='Directory to save/load trained pipelines')
    args = parser.parse_args()

    # Load data based on mode
    if args.mode == 'emnist':
        X_train, y_train, X_test, y_test = load_emnist_train_test(
            args.emnist_train_csv,
            args.emnist_test_csv,
            args.mapping_txt,
            quick=args.quick
        )
        model_name = 'emnist_lr_pipeline.joblib'
    else:
        X_train, X_test, y_train, y_test = load_tmnist_symbols(args.tmnist_csv)
        model_name = 'tmnist_symbols_lr_pipeline.joblib'

    # Determine PCA and model params
    n_components = 25 if args.quick else 100
    Cval = 1.0

    # Setup model path
    MODEL_PATH = os.path.join(args.model_dir, model_name)

    # Create or load pipeline (skip loading in quick mode)
    if os.path.exists(MODEL_PATH) and not args.quick:
        pipeline = joblib.load(MODEL_PATH)
        print(f"Loaded Logistic Regression model for {args.mode}.")
    else:
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=n_components)),
            ('lr', LogisticRegression(C=Cval, max_iter=1000, solver='lbfgs', n_jobs=-1))
        ])
        print(f"Training Logistic Regression model for {args.mode}...")
        pipeline.fit(X_train, y_train)
        if not args.quick:
            joblib.dump(pipeline, MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}.")

    # Scree plot: PCA explained variance
    pca = pipeline.named_steps['pca']
    evr = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(evr) + 1), evr, marker='o', linewidth=2)
    plt.title('Scree Plot: PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion matrix heatmap
    classes = sorted(np.unique(y_test), key=lambda c: ord(c))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix ({args.mode})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Logistic Regression coefficients heatmap
    lr = pipeline.named_steps['lr']
    coef = lr.coef_
    classes_lr = lr.classes_
    comp_labels = [f"PC{i+1}" for i in range(coef.shape[1])]
    plt.figure(figsize=(12, 8))
    sns.heatmap(coef, center=0, cmap='RdBu', xticklabels=comp_labels, yticklabels=classes_lr)
    plt.title("Logistic Regression Coefficients by Class and PCA Component")
    plt.xlabel("PCA Component")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.show()

    # ROC Curve (micro-average)
    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=classes_lr)
    y_score = pipeline.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'Micro-average ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
