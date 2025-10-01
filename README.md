# TakeHomeProject
# Census Income Classification & Segmentation Project

## Contents
- classification.py : Train & evaluate classification model (RandomForest with preprocessing).
- segmentation.py : Build segmentation (KMeans) and output cluster assignments + summary.
- census-bureau.columns : column header file (provided).
- census-bureau.data : dataset file (provided).
- PROJECT_REPORT.md : Project report for the client.
- rf_pipeline.joblib : (produced by classification script; contains preprocessing + model)
- kmeans_pipeline.joblib : (produced by segmentation script; contains preprocessor + kmeans)

## Environment & Dependencies
Recommended: Python 3.8+.

Install required packages:

```bash
python -m pip install -r requirements.txt
