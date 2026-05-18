"""
Credit Card Fraud Detection - Script d'Entraînement
====================================================
Dataset : credit_card_fraud.csv (339 607 transactions, ~0.5% de fraudes)

Étapes :
  1. Chargement & exploration
  2. Feature Engineering
  3. Préprocessing (encodage, normalisation)
  4. Rééchantillonnage SMOTE (déséquilibre de classes)
  5. Entraînement Random Forest + XGBoost
  6. Évaluation (classification report, ROC-AUC, confusion matrix)
  7. Sauvegarde du modèle + pipeline

Usage :
  python train_model.py --data credit_card_fraud.csv --output model/
"""

import argparse
import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost non disponible. Installez-le avec : pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imbalanced-learn non disponible. Installez avec : pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Chargement
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  CHARGEMENT DES DONNÉES")
    print(f"{'='*55}")
    df = pd.read_csv(path)
    print(f"  Lignes      : {len(df):,}")
    print(f"  Colonnes    : {df.shape[1]}")
    print(f"  Fraudes     : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"  Légitimes   : {(df['is_fraud']==0).sum():,}")
    return df


# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  FEATURE ENGINEERING")
    print(f"{'='*55}")

    df = df.copy()

    # Parsing date/heure
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]       = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"]      = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Âge du titulaire
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # Distance entre le titulaire et le marchand (formule de Haversine simplifiée)
    R = 6371
    lat1, lat2 = np.radians(df["lat"]), np.radians(df["merch_lat"])
    lon1, lon2 = np.radians(df["long"]), np.radians(df["merch_long"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df["distance_km"] = R * 2 * np.arcsin(np.sqrt(a))

    # Log-montant (réduire l'asymétrie)
    df["log_amt"] = np.log1p(df["amt"])

    print(f"  Nouvelles features : hour, day_of_week, month, is_weekend,")
    print(f"                       is_night, age, distance_km, log_amt")
    return df


# ─────────────────────────────────────────────
# 3. Préprocessing
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    "amt", "log_amt", "lat", "long", "city_pop",
    "merch_lat", "merch_long", "distance_km",
    "hour", "day_of_week", "month", "is_weekend", "is_night", "age"
]

CATEGORICAL_FEATURES = ["category", "state"]

def preprocess(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  PRÉPROCESSING")
    print(f"{'='*55}")

    df = df.copy()

    # Encodage des variables catégorielles
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES]
    X = df[feature_cols].fillna(0)
    y = df["is_fraud"]

    print(f"  Features utilisées : {len(feature_cols)}")
    print(f"  {feature_cols}")

    return X, y, encoders, feature_cols


# ─────────────────────────────────────────────
# 4. Split + SMOTE
# ─────────────────────────────────────────────

def split_and_resample(X, y, test_size=0.2, random_state=42):
    print(f"\n{'='*55}")
    print("  SPLIT & RÉÉCHANTILLONNAGE")
    print(f"{'='*55}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if HAS_SMOTE:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train_sc, y_train)
        print(f"  Après SMOTE → fraudes: {y_res.sum():,} / légitimes: {(y_res==0).sum():,}")
    else:
        X_res, y_res = X_train_sc, y_train
        print("  [WARN] SMOTE ignoré — imbalanced-learn non installé")

    return X_train_sc, X_test_sc, X_res, y_res, y_test, scaler


# ─────────────────────────────────────────────
# 5. Entraînement
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    print(f"\n{'='*55}")
    print("  ENTRAÎNEMENT DES MODÈLES")
    print(f"{'='*55}")

    models = {}

    # Random Forest
    print("  [1/2] Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf
    print("       ✓ Terminé")

    # XGBoost
    if HAS_XGB:
        print("  [2/2] XGBoost ...")
        scale_pos = int((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
        print("       ✓ Terminé")

    return models


# ─────────────────────────────────────────────
# 6. Évaluation
# ─────────────────────────────────────────────

def evaluate(models, X_test, y_test, output_dir):
    print(f"\n{'='*55}")
    print("  ÉVALUATION")
    print(f"{'='*55}")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    fig, axes = plt.subplots(len(models), 3, figsize=(18, 6 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)

        print(f"\n  ── {name.upper()} ──")
        print(f"  ROC-AUC       : {roc_auc:.4f}")
        print(f"  Avg Precision : {avg_prec:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Légitime", "Fraude"]))

        results[name] = {"roc_auc": roc_auc, "avg_precision": avg_prec}

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i][0],
                    xticklabels=["Légitime", "Fraude"],
                    yticklabels=["Légitime", "Fraude"])
        axes[i][0].set_title(f"{name} — Confusion Matrix")
        axes[i][0].set_ylabel("Réel")
        axes[i][0].set_xlabel("Prédit")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[i][1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="steelblue", lw=2)
        axes[i][1].plot([0,1],[0,1],"--", color="gray")
        axes[i][1].set_title(f"{name} — Courbe ROC")
        axes[i][1].set_xlabel("Taux de Faux Positifs")
        axes[i][1].set_ylabel("Taux de Vrais Positifs")
        axes[i][1].legend()

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        axes[i][2].plot(rec, prec, color="darkorange", lw=2,
                        label=f"AP = {avg_prec:.3f}")
        axes[i][2].set_title(f"{name} — Precision-Recall")
        axes[i][2].set_xlabel("Recall")
        axes[i][2].set_ylabel("Precision")
        axes[i][2].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_curves.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\n  📊 Graphiques sauvegardés → {plot_path}")
    return results


# ─────────────────────────────────────────────
# 7. Sauvegarde
# ─────────────────────────────────────────────

def save_artifacts(models, scaler, encoders, feature_cols, output_dir):
    print(f"\n{'='*55}")
    print("  SAUVEGARDE")
    print(f"{'='*55}")

    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        path = os.path.join(output_dir, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"  ✓ {name}.pkl")

    joblib.dump(scaler,   os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(output_dir, "encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(output_dir, "feature_cols.pkl"))
    print(f"  ✓ scaler.pkl  |  encoders.pkl  |  feature_cols.pkl")
    print(f"\n  Tous les artefacts → {output_dir}/")


# ─────────────────────────────────────────────
# 8. Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(models, feature_cols, output_dir):
    for name, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        imp.head(15).plot(kind="barh", ax=ax, color="steelblue")
        ax.invert_yaxis()
        ax.set_title(f"Top 15 Features — {name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        path = os.path.join(output_dir, f"feature_importance_{name}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  📊 Feature importance → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud — Entraînement")
    parser.add_argument("--data",   default="credit_card_fraud.csv", help="Chemin du CSV")
    parser.add_argument("--output", default="model/",                help="Dossier de sortie")
    args = parser.parse_args()

    # Pipeline complet
    df       = load_data(args.data)
    df       = feature_engineering(df)
    X, y, encoders, feature_cols = preprocess(df)

    X_train_sc, X_test_sc, X_res, y_res, y_test, scaler = split_and_resample(X, y)

    models   = train_models(X_res, y_res)
    results  = evaluate(models, X_test_sc, y_test, args.output)
    plot_feature_importance(models, feature_cols, args.output)
    save_artifacts(models, scaler, encoders, feature_cols, args.output)

    print(f"\n{'='*55}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*55}")
    for name, r in results.items():
        print(f"  {name:20s}  ROC-AUC={r['roc_auc']:.4f}  AP={r['avg_precision']:.4f}")
    print(f"\n  ✅ Entraînement terminé !\n")


if __name__ == "__main__":
    main()
