# Client: Retail Marketing (Walmart)
# Project: Income Classification and Customer Segmentation
# Prepared by: Ningjiang Huang
## =========================================================
# Summary

## I. Objective:
(1) Predict whether a person has income > $50K using 40 demographic/employment features.
(2) Create a segmentation model for marketing use.

## II. Deliverables:
(1) Trained classifier with evaluation and saved pipeline.
(2) Segmentation using KMeans with cluster summaries and assignments.
(3) Recommendations for how to use both models for marketing.

## III. Key findings (high level):
(1) The classification pipeline with RandomForest (preprocessing included) achieves reasonable discriminatory performance on hold-out data (accuracy / precision / recall / AUC reported by the script).
(2) Clustering reveals distinct groups that can be leveraged for targeted campaigns (e.g., clusters separating by age/education/occupation/industry).

# Data

## I. Source:
Weighted census data extracted from the 1994 and 1995 Current Population Surveys (U.S. Census Bureau). Provided files: census-bureau.data and census-bureau.columns.

## II. Structure:
40 demographic and employment related variables, plus a sample weight and an income label (<=50K or >50K). Each row represents an observation.

# Exploration & Quality Checks

I performed the following exploratory steps (implemented in the scripts):
(1) Read the header file census-bureau.columns to obtain column names, then loaded the CSV.
(2) Examined the label column values — labels sometimes have punctuation (e.g., - 50000.). Implemented robust label parsing to map to binary target.
(3) Looked for a sample weight column (keywords "weight" or "fnlwgt") and use it when present.
(4) Checked for missing values and common placeholder values (?, ?) and treated them as NA.

# Data Preprocessing Decisions

## I. Label cleaning
Converted textual label variants to binary: >50k -> 1, <=50k or - 50000. -> 0.

## II. Feature typing
Automatically inferred numeric vs categorical features by attempting numeric coercion on a sample of values.
Reasoning: Avoid assumptions about exact variable types because the provided header uses descriptive names.

## III. Imputation
(1) Numeric: median imputation.
(2) Categorical: most frequent (mode) imputation.
(3) Rationale: Median is robust to outliers; mode preserves common categories.

## IV. Encoding and scaling
(1) Categorical variables: One-Hot Encoding (handle_unknown='ignore').
(2) Numeric variables: Standard scaling (zero mean, unit variance).
(3) Rationale: One-hot avoids imposing numeric ordinality on categories; scaling helps tree-based or distance-based algorithms and PCA.

## V. Sample weights
When a weight column is detected, it is used during classifier training and evaluation to reflect population sampling.

# Modeling — Classification

## I. Model selected:
RandomForestClassifier (sklearn), wrapped in a pipeline with preprocessing.
Reasoning: Robust baseline for tabular mixed-type data, handles nonlinearities and interactions, relatively robust to outliers and missingness after imputation, produces feature importance for interpretation. Good trade-off between performance and interpretability for initial deployment.

## II. Training strategy:
(1) 80/20 stratified train/test split by target.
(2) When weight column available, training uses sample weights so model optimizes with respect to population representation.
(3) No heavy hyperparameter tuning by default (configurable in code). Default n_estimators=200.

## III. Evaluation:
(1) Metrics computed: Accuracy, Precision, Recall, F1, ROC-AUC.
(2) Weighted evaluation (if weight column is available) to reflect sampling design.
(3) The script prints these metrics on the hold-out test set. (See output from running classification.py.)

## IV. Interpretability:
Feature importance available via RandomForest. For marketing use, we can map the most important features (e.g., education, occupation, hours worked, age) to campaign targeting rules.

## V. Model Limitations & Next steps:
(1) The RandomForest is a nonparametric model — to convert to scoring rules for production you might add a simple logistic regression trained on top of the most important features for easier explanations.
(2) Consider stratified cross-validation with grouped sampling and hyperparameter tuning (GridSearchCV/RandomizedSearchCV) to optimize performance.
(3) Consider fairness checks — ensure model does not systematically discriminate against protected groups (race, sex, etc.). If adverse effects are found, consider fairness-aware adjustments.

# Modeling — Segmentation

## I. Goal: Create customer segments for marketing strategy (not supervised by income).

## II. Approach:
(1)Preprocess numeric and categorical features similar to the classifier.
(2)Optionally reduce dimensionality with PCA to capture main variance directions.
(3)Apply KMeans clustering (configurable n_clusters).
(4)Evaluate clusters by silhouette score and inspect cluster sizes and top features per cluster.

## III. Interpretation:
For each cluster we provide:
(1) Size (number of rows).
(2) Top contributing features (approximated by the largest absolute values in cluster centroids in feature space).

## IV. Use-cases:
(1) Define tailored offer sets (e.g., financial products targeted to clusters with high-income probability and age/occupation combos).
(2) Optimize marketing spend by selecting clusters with high expected ROI (combine with classifier output).

# Business Recommendations

## I. Two-stage approach for targeted marketing:
(1) Use classifier to identify likely high-income individuals (>50K).
(2) Within likely high-income group, use segmentation to tailor offers by lifestyle/industry/age clusters.

## II. Campaign prioritization:
(1) Assign budgets by cluster expected conversion and cluster size.
(2) For clusters with similar incomes but distinct features (e.g., young professionals vs older managers), create different creatives and channels.

## III. Deploy conservatively:
(1) Start with pilot campaigns on a few clusters, measure lift, then expand.
(2) Monitor model drift — dataset is from 1994-95 and behaviors change; retrain periodically with recent data.

## IV. Ethical & Compliance:
(1) Evaluate for bias across protected attributes (race, sex, citizenship).
(2) Avoid using attributes that are protected in a discriminatory fashion; consult legal/compliance.

# Implementation & Production Notes
(1) The pipeline is saved via joblib and can be loaded to score new observations.
(2) For production scoring, ensure input features are standardized to the same names and types as in the training set.
(3) If using sample weights for training, ensure downstream scoring interprets predictions correctly at population scale.

# Limitations
(1) The dataset is historical; real-world deployment should use up-to-date data.
(2) No heavy hyperparameter tuning or ensemble stacking was performed in this baseline.
(3) Clustering is unsupervised; cluster meaning should be validated with domain experts and small A/B tests.

# References
(1) UCI Machine Learning Repository — Adult Data Set (Census Income) — common baseline for income prediction.
(2) scikit-learn documentation — Pipelines, ColumnTransformer, RandomForest, KMeans.
(3) “An Introduction to Statistical Learning” — for supervised learning and clustering background.


# Appendix — How to reproduce
Install Python packages: pandas numpy scikit-learn joblib
Place census-bureau.columns and census-bureau.data in working dir.

Run:

python classification.py

python segmentation.py --n-clusters 6
