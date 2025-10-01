#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Train and evaluate a classifier for the census dataset.

Usage:
    python classification.py --data-file census-bureau.data --cols-file census-bureau.columns

The script:
 - Loads column names from the cols file.
 - Loads the CSV data (no header) and assigns column names.
 - Cleans label column to binary (<=50k = 0, >50k = 1).
 - Identifies numeric and categorical columns automatically.
 - Builds a preprocessing pipeline (imputing + scaling for numeric, imputing + OneHot for categorical).
 - Trains a RandomForest classifier (with sample weights if a weight column exists).
 - Evaluates on a hold-out test set and prints metrics (accuracy, precision, recall, f1, roc_auc).
 - Saves the trained pipeline+model with joblib.
"""


# In[11]:


import argparse
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[12]:


def read_columns(cols_file):
    with open(cols_file, 'r', encoding='utf-8', errors='ignore') as f:
        return [line.strip() for line in f if line.strip() != '']

def load_data(data_file, cols):
    return pd.read_csv(data_file, header=None, names=cols, na_values=['?', ' ?'], skipinitialspace=True)

def clean_label(series):
    return series.apply(lambda x: 1 if "+" in str(x) else 0)

def auto_feature_types(df, label_col):
    features = [c for c in df.columns if c != label_col]
    num_cols, cat_cols = [], []
    for c in features:
        try:
            pd.to_numeric(df[c].dropna().iloc[:50], errors='raise')
            num_cols.append(c)
        except:
            cat_cols.append(c)
    return num_cols, cat_cols

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K"," >50K"], yticklabels=["<=50K"," >50K"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0,1],[0,1],'--', color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_feature_importances(model, feature_names, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(7,5))
        plt.barh(np.array(feature_names)[idx], importances[idx])
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.show()

def main(args):
    cols = read_columns(args.cols_file)
    df = load_data(args.data_file, cols)

    label_candidates = [c for c in df.columns if 'income' in c.lower() or 'label' in c.lower()]
    if not label_candidates:
        raise ValueError("No label column found (expected 'income' or 'label').")
    label_col = label_candidates[0]

    df['target'] = clean_label(df[label_col])
    X = df.drop(columns=[label_col, 'target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    num_cols, cat_cols = auto_feature_types(X, label_col)
    num_transform = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_transform = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preproc = ColumnTransformer([("num", num_transform, num_cols), ("cat", cat_transform, cat_cols)])

    clf = Pipeline([("preproc", preproc), ("rf", RandomForestClassifier(n_estimators=200, random_state=42))])
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print("\nEvaluation Results:")
    print("Accuracy :", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall   :", recall_score(y_test, preds))
    print("F1 Score :", f1_score(y_test, preds))
    print("ROC AUC  :", roc_auc_score(y_test, probs))

    # Visualizations
    plot_confusion_matrix(y_test, preds)
    plot_roc(y_test, probs)

    # Get feature names for plotting importances
    cat_features = list(clf.named_steps["preproc"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
    all_features = num_cols + cat_features
    plot_feature_importances(clf.named_steps["rf"], all_features)

    joblib.dump(clf, args.output_model)
    print(f"\nModel saved to {args.output_model}")


# In[13]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="census-bureau.data")
    parser.add_argument("--cols-file", default="census-bureau.columns")
    parser.add_argument("--output-model", default="rf_pipeline.joblib")

    if "ipykernel" in sys.modules:  # Jupyter
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    main(args)


# In[ ]:




