#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
segmentation.py

Run KMeans clustering with visualizations.
"""


# In[5]:


import argparse
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# In[6]:


def read_columns(cols_file):
    with open(cols_file, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip() != ""]

def load_data(data_file, cols):
    return pd.read_csv(data_file, header=None, names=cols, na_values=["?", " ?"], skipinitialspace=True)

def auto_feature_types(df):
    num_cols, cat_cols = [], []
    for c in df.columns:
        try:
            pd.to_numeric(df[c].dropna().iloc[:50], errors="raise")
            num_cols.append(c)
        except:
            cat_cols.append(c)
    return num_cols, cat_cols

def plot_cluster_sizes(clusters):
    plt.figure(figsize=(6,4))
    sns.countplot(x=clusters, palette="viridis")
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.show()

def plot_pca_clusters(X_proc, clusters):
    pca = PCA(n_components=2, random_state=17)
    X_pca = pca.fit_transform(X_proc)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="tab10", alpha=0.6)
    plt.title("Clusters (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.show()

def main(args):
    cols = read_columns(args.cols_file)
    df = load_data(args.data_file, cols)

    # Drop label if exists
    label_candidates = [c for c in df.columns if "income" in c.lower() or "label" in c.lower()]
    if label_candidates:
        df = df.drop(columns=[label_candidates[0]])

    num_cols, cat_cols = auto_feature_types(df)

    num_transform = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_transform = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preproc = ColumnTransformer([("num", num_transform, num_cols), ("cat", cat_transform, cat_cols)])

    X_proc = preproc.fit_transform(df)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_proc)

    df["cluster"] = clusters
    df.to_csv(args.out_clusters, index=False)

    sil = silhouette_score(X_proc, clusters)
    print("\nKMeans Segmentation:")
    print(f"Clusters: {args.n_clusters}")
    print(f"Silhouette Score: {sil:.3f}")

    # Visualizations
    plot_cluster_sizes(clusters)
    plot_pca_clusters(X_proc, clusters)

    joblib.dump({"preprocessor": preproc, "kmeans": kmeans}, args.out_model)
    print(f"\nCluster assignments saved to {args.out_clusters}")
    print(f"Segmentation model saved to {args.out_model}")


# In[7]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="census-bureau.data")
    parser.add_argument("--cols-file", default="census-bureau.columns")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--out-clusters", default="clusters.csv")
    parser.add_argument("--out-model", default="kmeans_pipeline.joblib")

    if "ipykernel" in sys.modules:  # Jupyter
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    main(args)


# In[ ]:




