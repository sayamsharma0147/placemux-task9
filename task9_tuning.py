"""
Task 9 — Hyperparameter Tuning
PlaceMux · Phase 1 · AI/ML Developer
=====================================================
WHAT THIS SCRIPT DOES:
  Systematically tunes the RandomForest hyperparameters from Task 8's pipeline
  using RandomizedSearchCV with 5-fold cross-validation, scored by F1 (Hard class).
  Confirms the gain holds on the held-out test set (never touched during search).

  Run with:
      python task9_tuning.py

WHY RandomizedSearchCV OVER GridSearchCV:
  Grid search on 4 hyperparameters with even modest ranges = hundreds of fits.
  Random search samples the space more efficiently — Bergstra & Bengio (2012)
  showed random search finds near-optimal settings in far fewer evaluations,
  especially when only 1-2 params actually matter. We run 30 iterations × 5 folds
  = 150 fits, which is tractable and covers the space well.

WHY NOT BAYESIAN (Optuna):
  Bayesian search shines on expensive models (XGBoost, neural nets). For RF on
  ~3000 rows, random search is fast enough and produces comparable results without
  the extra dependency.

HYPERPARAMETERS CHOSEN AND WHY:
  n_estimators    — more trees = lower variance, but diminishing returns after ~200
  max_depth       — primary bias/variance knob; deep trees overfit, shallow underfit
  min_samples_leaf— minimum samples at a leaf; higher = smoother, less overfit
  max_features    — fraction of features per split; controls diversity across trees
  These 4 are the highest-leverage params for RF. Others (criterion, bootstrap)
  rarely move the needle enough to justify the search cost.

DELIVERABLE:
  Tuned model with best config, CV results, and confirmed test-set gain over default.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, json, warnings, hashlib
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

SEED      = 42
OUT_DIR   = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

print("=" * 60)
print("TASK 9 — HYPERPARAMETER TUNING")
print("PlaceMux · Phase 1 · AI/ML Developer")
print("=" * 60)
print(f"Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ── STAGE 1: DATA LOADING (identical to Task 8) ───────────────────────────────
print("── STAGE 1: DATA LOADING ──")
files = [f for f in sorted(glob.glob("/mnt/user-data/uploads/formatted_*.xlsx"))
         if "DevOps" not in f]
data = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
print(f"  Rows loaded : {len(data)} from {len(files)} files")

REQUIRED = ["question_text","option_a","option_b","option_c","option_d",
            "domain","topic","difficulty_level"]
missing = [c for c in REQUIRED if c not in data.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
print(f"  Schema      : ✓\n")

# ── STAGE 2: TARGET + SPLIT ───────────────────────────────────────────────────
print("── STAGE 2: TARGET ENGINEERING & SPLIT ──")
MEDIAN_SPLIT = 42.0
data["label"] = (data["difficulty_level"] >= MEDIAN_SPLIT).astype(int)

X_meta = data[["question_text","option_a","option_b","option_c",
               "option_d","domain","topic","difficulty_level"]]
y = data["label"]

X_train_meta, X_temp, y_train, y_temp = train_test_split(
    X_meta, y, test_size=0.30, random_state=SEED, stratify=y)
X_val_meta, X_test_meta, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"  Train: {len(X_train_meta)} | Val: {len(X_val_meta)} | Test: {len(X_test_meta)}\n")

# ── STAGE 3: FEATURE ENGINEERING (leakage-safe, from Task 7/8) ───────────────
print("── STAGE 3: FEATURE ENGINEERING ──")

# Fit aggregate maps and encoders on TRAIN ONLY — same pattern as Tasks 7 & 8
train_domain_map = X_train_meta.groupby("domain")["difficulty_level"].mean().to_dict()
train_topic_map  = X_train_meta.groupby("topic")["difficulty_level"].mean().to_dict()
global_mean      = X_train_meta["difficulty_level"].mean()
le_domain        = LabelEncoder().fit(data["domain"].astype(str))
le_topic         = LabelEncoder().fit(data["topic"].astype(str))

def engineer_features(df_meta):
    """
    Derives 11 baseline features (Task 7 locked set) from raw question metadata.
    Uses frozen train-only lookup tables for aggregate features — no leakage.
    """
    df = df_meta.copy()
    df["q_len"]              = df["question_text"].str.len().fillna(0)
    df["q_word_count"]       = df["question_text"].str.split().str.len().fillna(0)
    opt_cols = ["option_a","option_b","option_c","option_d"]
    for col in opt_cols:
        df[f"{col[:5]}_len"] = df[col].str.len().fillna(0)
    len_cols = [f"{c[:5]}_len" for c in opt_cols]
    df["avg_opt_len"]        = df[len_cols].mean(axis=1)
    df["max_opt_len"]        = df[len_cols].max(axis=1)
    df["opt_len_range"]      = df[len_cols].max(axis=1) - df[len_cols].min(axis=1)
    df["total_opt_len"]      = df[len_cols].sum(axis=1)
    df["q_to_avg_opt_ratio"] = df["q_len"] / (df["avg_opt_len"] + 1)
    df["domain_avg_difficulty"] = df["domain"].map(train_domain_map).fillna(global_mean)
    df["topic_avg_difficulty"]  = df["topic"].map(train_topic_map).fillna(global_mean)
    df["domain_enc"]         = le_domain.transform(df["domain"].astype(str))
    df["topic_enc"]          = le_topic.transform(df["topic"].astype(str))
    return df

BASELINE_FEATURES = [
    "q_to_avg_opt_ratio","q_len","topic_avg_difficulty","max_opt_len",
    "avg_opt_len","opt_len_range","total_opt_len","q_word_count",
    "topic_enc","domain_avg_difficulty","domain_enc"
]
NUM_FEATURES = [f for f in BASELINE_FEATURES if f not in ["topic_enc","domain_enc"]]
CAT_FEATURES = ["topic_enc","domain_enc"]

X_train = engineer_features(X_train_meta)[BASELINE_FEATURES]
X_val   = engineer_features(X_val_meta)[BASELINE_FEATURES]
X_test  = engineer_features(X_test_meta)[BASELINE_FEATURES]
print(f"  Features    : {len(BASELINE_FEATURES)} (Task 7 locked baseline)\n")

# ── STAGE 4: DEFAULT BASELINE (Task 8 model) ──────────────────────────────────
# Establish the benchmark we must beat. Default RF with no tuning.
# We score at threshold=0.29 (Task 6 cost-justified threshold — kept fixed
# throughout tuning so we're comparing apples to apples).
print("── STAGE 4: DEFAULT BASELINE ──")
THRESH = 0.29

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", "passthrough",    CAT_FEATURES),
])

default_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=100, random_state=SEED, class_weight="balanced"))
])
default_pipeline.fit(X_train, y_train)

y_prob_default_val  = default_pipeline.predict_proba(X_val)[:, 1]
y_prob_default_test = default_pipeline.predict_proba(X_test)[:, 1]
y_pred_default_val  = (y_prob_default_val  >= THRESH).astype(int)
y_pred_default_test = (y_prob_default_test >= THRESH).astype(int)

default_val_f1   = f1_score(y_val,  y_pred_default_val)
default_test_f1  = f1_score(y_test, y_pred_default_test)
default_test_acc = accuracy_score(y_test, y_pred_default_test)

print(f"  Default config   : n_estimators=100, max_depth=None, min_samples_leaf=1, max_features='sqrt'")
print(f"  Val  F1 (Hard)   : {default_val_f1:.4f}")
print(f"  Test F1 (Hard)   : {default_test_f1:.4f}  ← benchmark to beat")
print(f"  Test Accuracy    : {default_test_acc:.4f}\n")

# ── STAGE 5: HYPERPARAMETER SEARCH ───────────────────────────────────────────
# Strategy: RandomizedSearchCV with 30 iterations × 5-fold StratifiedKFold.
# Scoring: f1 for the Hard class (class index 1) — matches business metric.
# Search space: 4 high-leverage RF hyperparameters only.
# Test set is NEVER touched during search — only train data with CV.
print("── STAGE 5: RANDOMIZED SEARCH (30 iter × 5-fold CV) ──")
print("  Scoring metric  : F1 (Hard class)")
print("  Search space:")

param_dist = {
    # n_estimators: more trees reduce variance but plateau around 200-300
    "classifier__n_estimators"   : [50, 100, 150, 200, 300],

    # max_depth: None = fully grown (overfits); shallow = high bias
    # Range 5-30 covers the bias/variance tradeoff zone for this dataset size
    "classifier__max_depth"      : [5, 10, 15, 20, 30, None],

    # min_samples_leaf: higher = smoother decision boundary, less overfit
    # 1 is RF default; 5-20 adds regularisation
    "classifier__min_samples_leaf": [1, 2, 5, 10, 20],

    # max_features: fraction of features per split
    # 'sqrt' is RF default; lower = more diverse trees (more regularisation)
    "classifier__max_features"   : ["sqrt", "log2", 0.5, 0.7],
}
for param, values in param_dist.items():
    print(f"    {param.replace('classifier__',''):<22}: {values}")

# StratifiedKFold ensures class balance in every fold
# We use train data only — val and test are held out entirely
cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Pipeline for search — same preprocessor, RF with class_weight fixed
search_pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", "passthrough",    CAT_FEATURES),
    ])),
    ("classifier", RandomForestClassifier(
        random_state=SEED, class_weight="balanced"))
])

search = RandomizedSearchCV(
    estimator  = search_pipeline,
    param_distributions = param_dist,
    n_iter     = 30,          # 30 random combinations
    cv         = cv_scheme,   # 5-fold stratified
    scoring    = "f1",        # F1 for Hard class (positive label = 1)
    n_jobs     = -1,          # use all available cores
    random_state = SEED,
    verbose    = 1,
    refit      = True         # refit best params on full train set
)

print(f"\n  Running search (30 × 5 = 150 fits)...")
search.fit(X_train, y_train)

best_params  = search.best_params_
best_cv_f1   = search.best_score_
best_estimator = search.best_estimator_

print(f"\n  ✓ Search complete")
print(f"  Best CV F1 (mean over 5 folds) : {best_cv_f1:.4f}")
print(f"  Best hyperparameters:")
for k, v in best_params.items():
    print(f"    {k.replace('classifier__',''):<22}: {v}")

# ── STAGE 6: EVALUATE TUNED MODEL ────────────────────────────────────────────
# Select by CV score (not val score) — val set is reserved for final comparison.
# Now confirm the CV gain holds on val AND test (both unseen during search).
print("\n── STAGE 6: EVALUATE TUNED MODEL ──")

y_prob_tuned_val  = best_estimator.predict_proba(X_val)[:, 1]
y_prob_tuned_test = best_estimator.predict_proba(X_test)[:, 1]
y_pred_tuned_val  = (y_prob_tuned_val  >= THRESH).astype(int)
y_pred_tuned_test = (y_prob_tuned_test >= THRESH).astype(int)

tuned_val_f1   = f1_score(y_val,  y_pred_tuned_val)
tuned_test_f1  = f1_score(y_test, y_pred_tuned_test)
tuned_test_acc = accuracy_score(y_test, y_pred_tuned_test)
tuned_test_rec = recall_score(y_test, y_pred_tuned_test)
tuned_test_pre = precision_score(y_test, y_pred_tuned_test)

gain_val  = tuned_val_f1  - default_val_f1
gain_test = tuned_test_f1 - default_test_f1

print(f"\n  {'Metric':<22} {'Default':>10} {'Tuned':>10} {'Gain':>10}")
print(f"  {'-'*54}")
print(f"  {'Val F1 (Hard)':<22} {default_val_f1:>10.4f} {tuned_val_f1:>10.4f} {gain_val:>+10.4f}")
print(f"  {'Test F1 (Hard)':<22} {default_test_f1:>10.4f} {tuned_test_f1:>10.4f} {gain_test:>+10.4f}")
print(f"  {'Test Accuracy':<22} {default_test_acc:>10.4f} {tuned_test_acc:>10.4f} {tuned_test_acc-default_test_acc:>+10.4f}")

print(f"\n  Full classification report (tuned, test set):")
report = classification_report(y_test, y_pred_tuned_test, target_names=["Easy","Hard"])
print("\n".join("    " + l for l in report.splitlines()))

# Show top 5 CV results for transparency
print(f"\n  Top 5 CV results from search:")
cv_results = pd.DataFrame(search.cv_results_)
top5 = cv_results.nlargest(5, "mean_test_score")[
    ["mean_test_score","std_test_score",
     "param_classifier__n_estimators","param_classifier__max_depth",
     "param_classifier__min_samples_leaf","param_classifier__max_features"]
].reset_index(drop=True)
print(top5.to_string(index=False))

# ── STAGE 7: SAVE ARTIFACTS ───────────────────────────────────────────────────
print("\n── STAGE 7: SAVING ARTIFACTS ──")

# Save tuned model
model_path = OUT_DIR / "task9_tuned_model.joblib"
joblib.dump(best_estimator, model_path)
print(f"  ✓ Tuned model   : {model_path}")

# Reproducibility hash
pred_hash = hashlib.sha256(y_pred_tuned_test.tobytes()).hexdigest()

# Experiment log
log = {
    "task"          : "Task 9 — Hyperparameter Tuning",
    "timestamp"     : datetime.now().isoformat(),
    "seed"          : SEED,
    "threshold"     : THRESH,
    "search_strategy": "RandomizedSearchCV",
    "n_iter"        : 30,
    "cv_folds"      : 5,
    "scoring_metric": "f1 (Hard class)",
    "param_space"   : {k: str(v) for k, v in param_dist.items()},
    "best_params"   : {k.replace("classifier__",""): str(v) for k, v in best_params.items()},
    "best_cv_f1"    : round(best_cv_f1, 4),
    "default": {
        "val_f1"   : round(default_val_f1, 4),
        "test_f1"  : round(default_test_f1, 4),
        "test_acc" : round(default_test_acc, 4),
    },
    "tuned": {
        "val_f1"       : round(tuned_val_f1, 4),
        "test_f1"      : round(tuned_test_f1, 4),
        "test_acc"     : round(tuned_test_acc, 4),
        "test_precision": round(tuned_test_pre, 4),
        "test_recall"  : round(tuned_test_rec, 4),
    },
    "gain": {
        "val_f1"  : round(gain_val, 4),
        "test_f1" : round(gain_test, 4),
    },
    "reproducibility_hash": pred_hash,
    "note": "Test set was never used during search. CV scored on train folds only."
}

log_path = OUT_DIR / "task9_experiment_log.json"
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)
print(f"  ✓ Experiment log: {log_path}")
print(f"  ✓ Repro hash    : {pred_hash[:16]}...")

# ── STAGE 8: PLOTS ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Task 9 — Hyperparameter Tuning · PlaceMux Phase 1",
             fontsize=12, fontweight="bold")

# Plot 1: CV score distribution across all 30 iterations
cv_f1_scores = cv_results["mean_test_score"].values
axes[0].hist(cv_f1_scores, bins=10, color="#1976D2", edgecolor="white", alpha=0.85)
axes[0].axvline(best_cv_f1, color="red", linestyle="--", lw=2,
                label=f"Best CV F1 = {best_cv_f1:.3f}")
axes[0].axvline(default_val_f1, color="orange", linestyle="--", lw=2,
                label=f"Default Val F1 = {default_val_f1:.3f}")
axes[0].set_xlabel("Mean CV F1 (Hard)")
axes[0].set_ylabel("Count")
axes[0].set_title("CV Score Distribution\n(30 random configurations)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Plot 2: Default vs Tuned on Val and Test
metrics_labels = ["Val F1", "Test F1", "Test Acc"]
default_scores = [default_val_f1, default_test_f1, default_test_acc]
tuned_scores   = [tuned_val_f1,   tuned_test_f1,   tuned_test_acc]
x = np.arange(len(metrics_labels))
w = 0.35
bars1 = axes[1].bar(x - w/2, default_scores, w, label="Default", color="#90CAF9", edgecolor="white")
bars2 = axes[1].bar(x + w/2, tuned_scores,   w, label="Tuned",   color="#1976D2", edgecolor="white")
for bar, val in zip(list(bars1) + list(bars2), default_scores + tuned_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics_labels)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Score")
axes[1].set_title("Default vs Tuned\nVal & Test Performance")
axes[1].legend()
axes[1].grid(True, axis="y", alpha=0.3)

# Plot 3: Confusion matrix of tuned model on test set
cm = confusion_matrix(y_test, y_pred_tuned_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["Easy","Hard"])
disp.plot(ax=axes[2], colorbar=False)
axes[2].set_title(f"Confusion Matrix — Tuned Model\n(Test set, threshold={THRESH})")

plt.tight_layout()
plot_path = OUT_DIR / "task9_tuning.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  ✓ Plot saved    : {plot_path}\n")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
gain_status = "✓ GAIN CONFIRMED" if gain_test > 0 else "✗ NO GAIN — check search space"
print("=" * 60)
print("✓ TASK 9 COMPLETE — TUNING SUMMARY")
print("=" * 60)
print(f"  Search          : RandomizedSearchCV, 30 iter, 5-fold CV")
print(f"  Scoring metric  : F1 (Hard class)")
print(f"  Best CV F1      : {best_cv_f1:.4f}")
print(f"  Default Test F1 : {default_test_f1:.4f}")
print(f"  Tuned  Test F1  : {tuned_test_f1:.4f}  ({gain_test:+.4f})")
print(f"  Test confirmed  : {gain_status}")
print(f"\n  Best config:")
for k, v in best_params.items():
    print(f"    {k.replace('classifier__',''):<22}: {v}")
print(f"\n  Artifacts:")
print(f"    task9_tuned_model.joblib    — best tuned Pipeline")
print(f"    task9_experiment_log.json   — full log + CV results + hash")
print(f"    task9_tuning.png            — CV distribution, comparison, confusion matrix")
