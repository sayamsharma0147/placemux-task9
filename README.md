Task 9 — Hyperparameter Tuning
PlaceMux · Altrodav Technologies · Phase 1 Industry Immersion
Objective
Systematically tune the RandomForest hyperparameters from Task 8's pipeline using cross-validation, confirm the gain holds on the held-out test set, and record the best configuration.
How to Run
bashpip install scikit-learn joblib matplotlib openpyxl pandas
python task9_tuning.py
Search Strategy
RandomizedSearchCV chosen over GridSearchCV:

Grid search across 4 parameters with even modest ranges = hundreds of fits. Random search samples the space more efficiently — Bergstra & Bengio (2012) showed it finds near-optimal settings in far fewer evaluations, especially when only 1-2 params actually drive the metric. We ran 30 iterations × 5 folds = 150 fits, which is tractable and covers the space well.
Bayesian search (Optuna) not used:

Bayesian search shines on expensive models (XGBoost, neural nets). For RF on ~3,000 rows, random search is fast enough and produces comparable results without the extra dependency.
Hyperparameters Searched
ParameterSearch RangeWhy it mattersn_estimators[50, 100, 150, 200, 300]More trees = lower variance; diminishing returns after ~200max_depth[5, 10, 15, 20, 30, None]Primary bias/variance knob — deep trees overfitmin_samples_leaf[1, 2, 5, 10, 20]Higher = smoother boundary, less overfitmax_features[sqrt, log2, 0.5, 0.7]Controls diversity across trees
Parameters not searched: criterion, bootstrap — rarely move the needle enough to justify the compute cost.
CV Scheme

StratifiedKFold(n_splits=5) — preserves class balance in every fold
Scored by f1 (Hard class, positive label = 1) — matches business metric
Test set never touched during search

Results
MetricDefaultTunedGainVal F1 (Hard)0.68420.6957+0.0114Test F1 (Hard)0.67540.6985+0.0231Test Accuracy0.58750.5792-0.0083
Accuracy dipped slightly — expected, since we optimised for F1 (Hard recall), not accuracy. The gain that matters held on test.
Best Configuration
ParameterDefaultBest Tunedn_estimators10050max_depthNone5min_samples_leaf15max_featuressqrtsqrt
Shallower trees (max_depth=5) with leaf regularisation (min_samples_leaf=5) won — default unlimited depth was overfitting. Fewer trees (50) were sufficient.
Top 5 CV Configurations
Mean CV F1Stdn_estimatorsmax_depthmin_samples_leafmax_features0.65920.0185055sqrt0.65790.0165052log20.65710.02315051sqrt0.65490.01850520.50.65120.022100550.7
Artifacts
FileDescriptiontask9_tuning.pyFull tuning script with detailed commentstask9_tuned_model.joblibBest tuned sklearn Pipelinetask9_experiment_log.jsonFull log — best params, CV scores, gain, repro hashtask9_tuning.pngCV distribution + Default vs Tuned comparison + confusion matrix
Load the Tuned Model
pythonimport joblib
pipeline = joblib.load("task9_tuned_model.joblib")
predictions = pipeline.predict(X_new)
Stack

Python 3.12, scikit-learn, joblib, pandas, matplotlib, openpyxl
