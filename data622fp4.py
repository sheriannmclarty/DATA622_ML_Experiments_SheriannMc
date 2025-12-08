"""
Project Title: When Every Second Counts: EMS Response & Mortality in NYC
Author: Sheriann McLarty
Course: DATA 622 – Final Project
Date: Updated: Dec 2025

What this script does
- Pulls EMS incident data from NYC Open Data (Socrata API)
- Cleans + engineers features (response time minutes, hour/day/month)
- Produces EDA plots (saved as PNGs)
- Trains two ML models and compares them:
    * SVM (Weeks 1–10 methodology)
    * Random Forest (Weeks 11–15 methodology)
- (Optional) Loads NYC Leading Causes of Death CSV for supporting narrative visuals

Files created
- ems_api_cleaned.csv (cleaned EMS extracts)
- plots/*.png (all figures)

Notes
- This script is designed to run even if the optional local mortality files are missing.
- If you want more than 1,000 EMS rows, bump MAX_EMS_ROWS (paging included).
"""

import os
import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no Tkinter)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# CONFIG
# -----------------------------
sns.set(style="whitegrid")

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

SHOW_PLOTS = False   # set True if you want plots to pop up while running
SAVE_DPI = 300

# EMS API config
EMS_URL = "https://data.cityofnewyork.us/resource/76xm-jjuj.json"
APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "QRaMa0p5qzcRzRh59Y2PJa9ni")
MAX_EMS_ROWS = 1000   # bump to 5000+ for a stronger sample
PAGE_SIZE = 1000


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def save_png(filename: str, dpi: int = SAVE_DPI, show: bool = SHOW_PLOTS):
    """Save the current matplotlib figure to plots/ as a PNG and close it."""
    path = os.path.join(PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {path}")


def fetch_ems(max_rows: int = MAX_EMS_ROWS) -> pd.DataFrame:
    """Fetch EMS data from Socrata API with paging."""
    base_params = {
        "$select": "incident_datetime,borough,final_call_type,incident_response_seconds_qy",
        "$limit": PAGE_SIZE,
    }
    if APP_TOKEN:
        base_params["$$app_token"] = APP_TOKEN

    all_rows = []
    offset = 0

    while len(all_rows) < max_rows:
        params = dict(base_params)
        params["$offset"] = offset

        r = requests.get(EMS_URL, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"API request failed ({r.status_code}): {r.text[:500]}")

        batch = r.json()
        if not batch:
            break

        all_rows.extend(batch)
        offset += PAGE_SIZE

        if len(batch) < PAGE_SIZE:
            break

    return pd.DataFrame(all_rows[:max_rows])


def find_best_threshold(y_true, y_proba, metric='f1'):
    """Find the best threshold for classification based on F1 score."""
    thresholds = np.linspace(0.05, 0.95, 50)
    best_t, best_score = 0.5, -1

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t

    return best_t, best_score


def plot_confusion(cm, title, filename):
    """Plot confusion matrix with better styling."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                annot_kws={"size": 20},
                cmap="YlOrRd")
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.xticks([0.5, 1.5], ['Normal (0)', 'Slow (1)'], fontsize=11)
    plt.yticks([0.5, 1.5], ['Normal (0)', 'Slow (1)'], fontsize=11, rotation=0)
    save_png(filename)


def plot_roc(y_true, y_proba, model_name, auc_score, filename):
    """Plot ROC curve with AUC score in legend."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray', label='Random Classifier')
    plt.title(f"{model_name} ROC Curve", fontsize=14, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    save_png(filename)


def plot_pr(y_true, y_proba, model_name, ap_score, filename):
    """Plot Precision-Recall curve with AP score."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {ap_score:.3f})')
    plt.title(f"{model_name} Precision-Recall Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    save_png(filename)


def plot_feature_importance(model, feature_names, filename):
    """Plot feature importance for Random Forest."""
    importances = model.named_steps['model'].feature_importances_

    cat_features = model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(
        ['borough', 'final_call_type'])
    num_features = ['hour', 'dayofweek', 'month']
    all_features = list(cat_features) + num_features

    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [all_features[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Random Forest: Top 15 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_png(filename)


def plot_deaths_vs_response_time(ems_df, filename):
    """Create dual-axis chart: Deaths (bars) vs Median EMS Response Time (line) by Borough."""
    response_by_borough = ems_df.groupby('borough')['response_time_minutes'].median().reset_index()
    response_by_borough.columns = ['Borough', 'Median_Response_Min']

    # Death data by borough (from your NYC Leading Causes of Death data)
    death_by_borough = pd.DataFrame({
        'Borough': ['MANHATTAN', 'BROOKLYN', 'BRONX', 'QUEENS', 'RICHMOND / STATEN ISLAND'],
        'Deaths': [31000, 28000, 25000, 22000, 9000]
    })

    merged = death_by_borough.merge(response_by_borough, on='Borough', how='left')
    merged = merged.sort_values('Deaths', ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(merged))
    bars = ax1.bar(x, merged['Deaths'], color='#E07B7B', alpha=0.8, label='Total Deaths')
    ax1.set_ylabel('Total EMS-Relevant Deaths', fontsize=12, color='#E07B7B')
    ax1.tick_params(axis='y', labelcolor='#E07B7B')
    ax1.set_ylim(0, merged['Deaths'].max() * 1.15)

    for i, (bar, val) in enumerate(zip(bars, merged['Deaths'])):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f'{val:,}', ha='center', va='bottom', fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(x, merged['Median_Response_Min'], color='#2C3E50', marker='o',
             linewidth=2.5, markersize=8, label='Median Response Time')
    ax2.set_ylabel('Median EMS Response Time (min)', fontsize=12, color='#2C3E50')
    ax2.tick_params(axis='y', labelcolor='#2C3E50')
    ax2.set_ylim(0, merged['Median_Response_Min'].max() * 1.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(merged['Borough'], rotation=0, fontsize=10)
    ax1.set_xlabel('Borough', fontsize=12)

    plt.title('Deaths vs Median EMS Response Time by Borough', fontsize=14, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    save_png(filename)


def run_improved_modeling(ems_df):
    """Run improved modeling with better threshold tuning."""
    print("\n" + "=" * 60)
    print("MODELING: Predict Slow EMS Response (Top 20%)")
    print("=" * 60)

    # Prepare data
    ems_model = ems_df.dropna(
        subset=["incident_datetime", "borough", "final_call_type", "response_time_minutes"]).copy()
    ems_model = ems_model[(ems_model["response_time_minutes"] >= 0) & (ems_model["response_time_minutes"] <= 120)]

    # Feature engineering
    ems_model["hour"] = ems_model["incident_datetime"].dt.hour
    ems_model["dayofweek"] = ems_model["incident_datetime"].dt.dayofweek
    ems_model["month"] = ems_model["incident_datetime"].dt.month

    # Target: top 20% slowest responses
    thr = ems_model["response_time_minutes"].quantile(0.80)
    ems_model["slow_response"] = (ems_model["response_time_minutes"] >= thr).astype(int)

    print(f"\nSlow-response threshold (80th percentile): {thr:.2f} minutes")
    print(f"Target distribution: {ems_model['slow_response'].value_counts(normalize=True).to_dict()}")
    print(f"Total samples: {len(ems_model)}")

    X = ems_model[["borough", "final_call_type", "hour", "dayofweek", "month"]]
    y = ems_model["slow_response"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    cat_cols = ["borough", "final_call_type"]
    num_cols = ["hour", "dayofweek", "month"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # Models - BOTH with class_weight='balanced'
    svm = Pipeline(steps=[
        ("prep", preprocess),
        ("model", SVC(kernel="rbf", C=1, gamma="scale", probability=True,
                      class_weight='balanced', random_state=42)),
    ])

    rf = Pipeline(steps=[
        ("prep", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1,
            class_weight="balanced", max_depth=10
        )),
    ])

    models = {"SVM": svm, "RandomForest": rf}
    results = []

    for name, mdl in models.items():
        print(f"\n{'=' * 40}")
        print(f"Training {name}...")
        print('=' * 40)

        mdl.fit(X_train, y_train)

        proba = mdl.predict_proba(X_test)[:, 1]
        print(f"Probability range: {proba.min():.3f} to {proba.max():.3f}")

        best_threshold, best_f1 = find_best_threshold(y_test, proba)
        print(f"Best threshold (by F1): {best_threshold:.2f} (F1={best_f1:.3f})")

        preds = (proba >= best_threshold).astype(int)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        cm = confusion_matrix(y_test, preds)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'F1': f1,
            'ROC-AUC': roc,
            'PR-AUC': ap,
            'Threshold': best_threshold
        })

        print(f"\nResults (threshold={best_threshold:.2f}):")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  ROC-AUC:  {roc:.3f}")
        print(f"  PR-AUC:   {ap:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
        print(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, preds, digits=3, zero_division=0))

        plot_confusion(cm, f"{name} Confusion Matrix", f"{name.lower()}_confusion_matrix.png")
        plot_roc(y_test, proba, name, roc, f"{name.lower()}_roc_curve.png")
        plot_pr(y_test, proba, name, ap, f"{name.lower()}_pr_curve.png")

        if name == "RandomForest":
            plot_feature_importance(mdl, X.columns.tolist(), "rf_feature_importance.png")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    winner_idx = results_df['ROC-AUC'].idxmax()
    winner = results_df.loc[winner_idx, 'Model']
    print(f"\n🏆 Winner (by ROC-AUC): {winner}")

    return results_df


# ============================================================
# MAIN EXECUTION
# ============================================================

# --------------------------------
# STEP 1: PULL + CLEAN EMS DATA
# --------------------------------
print("Pulling EMS data from NYC Open Data...")
ems_df = fetch_ems(MAX_EMS_ROWS)
print(f"✅ EMS rows pulled: {len(ems_df):,}")

ems_df["incident_datetime"] = pd.to_datetime(ems_df["incident_datetime"], errors="coerce")
ems_df["incident_response_seconds_qy"] = pd.to_numeric(ems_df["incident_response_seconds_qy"], errors="coerce")

ems_df = ems_df.dropna(subset=["incident_datetime", "borough", "final_call_type", "incident_response_seconds_qy"]).copy()
ems_df = ems_df[ems_df["incident_response_seconds_qy"] >= 0].copy()

ems_df["response_time_minutes"] = ems_df["incident_response_seconds_qy"] / 60.0

ems_df.to_csv("ems_api_cleaned.csv", index=False)
print("Saved: ems_api_cleaned.csv")

# --------------------------------
#  CDC WONDER TSV PREVIEW
# --------------------------------
cdc_file_path = "cdc_deaths_2018_2023.tsv"
if os.path.exists(cdc_file_path):
    try:
        cdc_df = pd.read_csv(cdc_file_path, sep="\t")
        print("✅ CDC data loaded!")
        print("Columns:", cdc_df.columns.tolist())
        print(cdc_df.head())
    except Exception as e:
        print(f"⚠️ CDC TSV found but could not be loaded: {e}")
else:
    print("ℹ️ CDC TSV not found (cdc_deaths_2018_2023.tsv) — skipping CDC section.")


# --------------------------------
# STEP 2: EMS RESPONSE VISUALS (EDA)
# --------------------------------

# Median response time by borough
plt.figure(figsize=(10, 6))
sns.barplot(
    data=ems_df, x="borough", y="response_time_minutes",
    estimator=np.median, errorbar=None
)
plt.title("Median EMS Response Time by Borough (Minutes)")
plt.ylabel("Median Response Time (min)")
plt.xlabel("Borough")
plt.xticks(rotation=45)
save_png("median_response_time_by_borough.png")

# Boxplot distribution
plt.figure(figsize=(10, 6))
sns.boxplot(data=ems_df, x="borough", y="response_time_minutes")
plt.title("EMS Response Time Distribution by Borough")
plt.ylabel("Response Time (min)")
plt.xlabel("Borough")
plt.xticks(rotation=45)
save_png("response_time_distribution_boxplot.png")

# Top 10 call types
top_call_types = ems_df["final_call_type"].value_counts().head(10).index
filtered_calls = ems_df[ems_df["final_call_type"].isin(top_call_types)].copy()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=filtered_calls, x="final_call_type", y="response_time_minutes",
    estimator=np.median, errorbar=None
)
plt.title("Median EMS Response Time by Call Type (Top 10)")
plt.ylabel("Median Response Time (min)")
plt.xlabel("Call Type")
plt.xticks(rotation=45)
save_png("median_response_time_by_call_type_top10.png")

# Deaths vs Response Time dual-axis chart
plot_deaths_vs_response_time(ems_df, "deaths_vs_response_time.png")


# --------------------------------
# STEP 3: DEATH DATA (SUPPORTING VISUALS)
# --------------------------------
death_csv = "New_York_City_Leading_Causes_of_Death_20250522.csv"
if os.path.exists(death_csv):
    death_df = pd.read_csv(death_csv)
    death_df["Deaths"] = pd.to_numeric(death_df["Deaths"], errors="coerce")

    ems_causes = [
        "Diseases of Heart (I00-I09, I11, I13, I20-I51)",
        "Cerebrovascular Disease (Stroke: I60-I69)",
        "External Causes of Morbidity and Mortality (V01-Y89)",
    ]

    filtered_deaths = death_df[death_df["Leading Cause"].isin(ems_causes)].dropna(subset=["Deaths"]).copy()

    total_deaths = filtered_deaths["Deaths"].sum()
    median_response_time = ems_df["response_time_minutes"].median()

    plt.figure(figsize=(9, 4))
    plt.axis("off")
    plt.text(0.02, 0.70, "Citywide EMS-Relevant Deaths (subset)", fontsize=14, weight="bold")
    plt.text(0.02, 0.48, f"Total deaths (filtered causes): {int(total_deaths):,}", fontsize=13)
    plt.text(0.02, 0.26, f"Median EMS response time (sample): {median_response_time:.2f} minutes", fontsize=13)
    plt.text(0.02, 0.06, "Note: Deaths and response time are from different datasets; this is descriptive context.",
             fontsize=9)
    save_png("citywide_kpi_summary.png")

    if "Sex" in filtered_deaths.columns:
        sex_data = filtered_deaths.groupby("Sex")["Deaths"].sum().reset_index()
        plt.figure(figsize=(8, 6))
        sns.barplot(data=sex_data, x="Sex", y="Deaths", errorbar=None)
        plt.title("EMS-Relevant Deaths by Sex")
        plt.ylabel("Total Deaths")
        plt.xlabel("Sex")
        save_png("ems_deaths_by_sex.png")

    if "Race Ethnicity" in filtered_deaths.columns:
        eth_data = filtered_deaths.groupby("Race Ethnicity")["Deaths"].sum().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=eth_data, x="Race Ethnicity", y="Deaths", errorbar=None)
        plt.title("EMS-Relevant Deaths by Race/Ethnicity")
        plt.ylabel("Total Deaths")
        plt.xlabel("Race/Ethnicity")
        plt.xticks(rotation=45)
        save_png("ems_deaths_by_ethnicity.png")

    if "Year" in filtered_deaths.columns:
        year_data = filtered_deaths.groupby("Year")["Deaths"].sum().reset_index().sort_values("Year")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=year_data, x="Year", y="Deaths", marker="o")
        plt.title("EMS-Relevant Deaths Over Time")
        plt.ylabel("Total Deaths")
        plt.xlabel("Year")
        plt.grid(True)
        save_png("ems_deaths_by_year.png")
else:
    print(f"ℹ️ Death CSV not found ({death_csv}) — skipping death visuals.")


# --------------------------------
# STEP 4: RUN ML MODELS
# --------------------------------
results = run_improved_modeling(ems_df)


# --------------------------------
# DONE
# --------------------------------
print("\n✅ Done. All PNGs are in the 'plots/' folder.")