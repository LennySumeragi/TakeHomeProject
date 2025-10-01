Client: Retail Marketing (Walmart)
Project: Income Classification and Customer Segmentation
Prepared by: Ningjiang Huang

Executive Summary

Objective:
(1) Predict whether a person has income > $50K using 40 demographic/employment features.
(2) Create a segmentation model for marketing use.

Deliverables:
Trained classifier with evaluation and saved pipeline.
Segmentation using KMeans with cluster summaries and assignments.
Recommendations for how to use both models for marketing.

Key findings (high level):
The classification pipeline with RandomForest (preprocessing included) achieves reasonable discriminatory performance on hold-out data (accuracy / precision / recall / AUC reported by the script).
Clustering reveals distinct groups that can be leveraged for targeted campaigns (e.g., clusters separating by age/education/occupation/industry).

Data
Source:
Weighted census data extracted from the 1994 and 1995 Current Population Surveys (U.S. Census Bureau). Provided files: census-bureau.data and census-bureau.columns.

Structure:
40 demographic and employment related variables, plus a sample weight and an income label (<=50K or >50K). Each row represents an observation.

Exploration & Quality Checks
I performed the following exploratory steps (implemented in the scripts):
Read the header file census-bureau.columns to obtain column names, then loaded the CSV.
Examined the label column values — labels sometimes have punctuation (e.g., - 50000.). Implemented robust label parsing to map to binary target.
Looked for a sample weight column (keywords "weight" or "fnlwgt") and use it when present.
Checked for missing values and common placeholder values (?, ?) and treated them as NA.

Data Preprocessing Decisions

Label cleaning
Converted textual label variants to binary: >50k -> 1, <=50k or - 50000. -> 0.

Feature typing
Automatically inferred numeric vs categorical features by attempting numeric coercion on a sample of values.

Reasoning: Avoid assumptions about exact variable types because the provided header uses descriptive names.

Imputation
Numeric: median imputation.
Categorical: most frequent (mode) imputation.
Rationale: Median is robust to outliers; mode preserves common categories.

Encoding and scaling
Categorical variables: One-Hot Encoding (handle_unknown='ignore').
Numeric variables: Standard scaling (zero mean, unit variance).
Rationale: One-hot avoids imposing numeric ordinality on categories; scaling helps tree-based or distance-based algorithms and PCA.

Sample weights
When a weight column is detected, it is used during classifier training and evaluation to reflect population sampling.

Modeling — Classification
Model selected:
RandomForestClassifier (sklearn), wrapped in a pipeline with preprocessing.
Reasoning: Robust baseline for tabular mixed-type data, handles nonlinearities and interactions, relatively robust to outliers and missingness after imputation, produces feature importance for interpretation. Good trade-off between performance and interpretability for initial deployment.

Training strategy:
80/20 stratified train/test split by target.
When weight column available, training uses sample weights so model optimizes with respect to population representation.
No heavy hyperparameter tuning by default (configurable in code). Default n_estimators=200.

Evaluation:
Metrics computed: Accuracy, Precision, Recall, F1, ROC-AUC.
Weighted evaluation (if weight column is available) to reflect sampling design.
The script prints these metrics on the hold-out test set. (See output from running classification.py.)

Interpretability:
Feature importance available via RandomForest. For marketing use, we can map the most important features (e.g., education, occupation, hours worked, age) to campaign targeting rules.

Model Limitations & Next steps:
The RandomForest is a nonparametric model — to convert to scoring rules for production you might add a simple logistic regression trained on top of the most important features for easier explanations.
Consider stratified cross-validation with grouped sampling and hyperparameter tuning (GridSearchCV/RandomizedSearchCV) to optimize performance.
Consider fairness checks — ensure model does not systematically discriminate against protected groups (race, sex, etc.). If adverse effects are found, consider fairness-aware adjustments.

Modeling — Segmentation

Goal: Create customer segments for marketing strategy (not supervised by income).

Approach:
Preprocess numeric and categorical features similar to the classifier.
Optionally reduce dimensionality with PCA to capture main variance directions.
Apply KMeans clustering (configurable n_clusters).
Evaluate clusters by silhouette score and inspect cluster sizes and top features per cluster.

Interpretation:
For each cluster we provide:
Size (number of rows).
Top contributing features (approximated by the largest absolute values in cluster centroids in feature space).

Use-cases:
Define tailored offer sets (e.g., financial products targeted to clusters with high-income probability and age/occupation combos).
Optimize marketing spend by selecting clusters with high expected ROI (combine with classifier output).

Business Recommendations

Two-stage approach for targeted marketing:
Use classifier to identify likely high-income individuals (>50K).
Within likely high-income group, use segmentation to tailor offers by lifestyle/industry/age clusters.

Campaign prioritization:
Assign budgets by cluster expected conversion and cluster size.
For clusters with similar incomes but distinct features (e.g., young professionals vs older managers), create different creatives and channels.

Deploy conservatively:
Start with pilot campaigns on a few clusters, measure lift, then expand.
Monitor model drift — dataset is from 1994-95 and behaviors change; retrain periodically with recent data.

Ethical & Compliance:
Evaluate for bias across protected attributes (race, sex, citizenship).
Avoid using attributes that are protected in a discriminatory fashion; consult legal/compliance.

Implementation & Production Notes
The pipeline is saved via joblib and can be loaded to score new observations.
For production scoring, ensure input features are standardized to the same names and types as in the training set.
If using sample weights for training, ensure downstream scoring interprets predictions correctly at population scale.

Limitations
The dataset is historical; real-world deployment should use up-to-date data.
No heavy hyperparameter tuning or ensemble stacking was performed in this baseline.
Clustering is unsupervised; cluster meaning should be validated with domain experts and small A/B tests.

References
UCI Machine Learning Repository — Adult Data Set (Census Income) — common baseline for income prediction.
scikit-learn documentation — Pipelines, ColumnTransformer, RandomForest, KMeans.
“An Introduction to Statistical Learning” — for supervised learning and clustering background.


Appendix — How to reproduce
Install Python packages: pandas numpy scikit-learn joblib
Place census-bureau.columns and census-bureau.data in working dir.

Run:

python classification.py

python segmentation.py --n-clusters 6
