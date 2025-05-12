import os
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
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
        description="Train/test RBF-SVM on EMNIST letters+digits or TMNIST symbols"
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
        model_name = 'emnist_svm_pipeline.joblib'
    else:
        X_train, X_test, y_train, y_test = load_tmnist_symbols(args.tmnist_csv)
        model_name = 'tmnist_symbols_svm_pipeline.joblib'

    # Determine PCA and SVM params
    if args.quick:
        n_components = 25
        Cval = 2
        gamma_val = 0.03
    else:
        n_components = 100
        Cval = 10
        gamma_val = 0.05

    # Setup model path
    MODEL_PATH = os.path.join(args.model_dir, model_name)

    # Create or load pipeline (skip loading in quick mode)
    if os.path.exists(MODEL_PATH) and not args.quick:
        pipeline = joblib.load(MODEL_PATH)
        print(f"Loaded model for {args.mode}.")
    else:
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=n_components)),
            ('svm', SVC(kernel='rbf', C=Cval, gamma=gamma_val))
        ])
        print(f"Training new model for {args.mode}...")
        pipeline.fit(X_train, y_train)
        if not args.quick:
            joblib.dump(pipeline, MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}.")

    # Scree plot after PCA
    pca = pipeline.named_steps['pca']
    evr = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(evr) + 1), evr, 'o-', linewidth=2)
    plt.title('Scree Plot: PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate and classification report
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

    # F1-score bar chart
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=report_df.index[:-3], y=report_df['f1-score'][:-3])
    plt.xticks(rotation=90)
    plt.title(f"F1-Score by Class ({args.mode})")
    plt.xlabel("Class")
    plt.ylabel("F1-Score")
    plt.tight_layout()
    plt.show()



    # Decision boundary plot on first 2 PCA components, maybe works takes a long time to run
   # scaler = pipeline.named_steps['scaler']
   # X_train_scaled = scaler.transform(X_train)
   # X_test_scaled = scaler.transform(X_test)
   # pca2 = PCA(n_components=2)
   # X_train_pca2 = pca2.fit_transform(X_train_scaled)
   # X_test_pca2 = pca2.transform(X_test_scaled)
   # svm_2d = SVC(kernel='rbf', C=Cval, gamma=gamma_val)
   # svm_2d.fit(X_train_pca2, y_train)

    # create meshgrid
   # x_min, x_max = X_test_pca2[:,0].min() - 1, X_test_pca2[:,0].max() + 1
   # y_min, y_max = X_test_pca2[:,1].min() - 1, X_test_pca2[:,1].max() + 1
   # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
   # Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
   # Z = Z.reshape(xx.shape)

   # plt.figure(figsize=(8, 6))
   # plt.contourf(xx, yy, Z, alpha=0.3)
   # scatter = plt.scatter(X_test_pca2[:,0], X_test_pca2[:,1], c=[classes.index(c) for c in y_test],
   #                       edgecolor='k', cmap='tab10')
   # plt.title('Decision Boundary on First 2 PCA Components')
   # plt.xlabel('PC1')
   # plt.ylabel('PC2')
   # plt.legend(handles=scatter.legend_elements()[0], labels=classes, bbox_to_anchor=(1.05, 1), loc='upper left')
   # plt.tight_layout()
   # plt.show()


if __name__ == '__main__':
    main()
