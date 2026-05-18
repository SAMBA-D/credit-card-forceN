"""
Credit Card Fraud Detection — Interface Streamlit
==================================================
Lance avec :  streamlit run app.py

Prérequis :
  pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn plotly
  pip install xgboost imbalanced-learn   # optionnel mais recommandé

Structure attendue après entraînement :
  model/
    random_forest.pkl
    xgboost.pkl        (si disponible)
    scaler.pkl
    encoders.pkl
    feature_cols.pkl
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config page
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e63946, #457b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #e63946;
    }
    .fraud-badge {
        background: #e63946;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .legit-badge {
        background: #2a9d8f;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Chargement modèle
# ─────────────────────────────────────────────

MODEL_DIR = "model"

@st.cache_resource
def load_model_artifacts():
    """Charge les artefacts du modèle entraîné."""
    try:
        scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        encoders     = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
        feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

        # Choisir le meilleur modèle disponible
        for model_name in ["xgboost", "random_forest"]:
            path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            if os.path.exists(path):
                model = joblib.load(path)
                return model, scaler, encoders, feature_cols, model_name

        return None, None, None, None, None
    except Exception as e:
        return None, None, None, None, str(e)

# ─────────────────────────────────────────────
# Feature Engineering (miroir du script d'entraînement)
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    "amt", "log_amt", "lat", "long", "city_pop",
    "merch_lat", "merch_long", "distance_km",
    "hour", "day_of_week", "month", "is_weekend", "is_night", "age"
]
CATEGORICAL_FEATURES = ["category", "state"]

CATEGORIES = [
    "gas_transport", "grocery_net", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel", "entertainment", "food_dining"
]

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL",
    "IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT",
    "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
    "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]

def compute_features(row: dict, encoders: dict) -> pd.DataFrame:
    """Calcul des features à partir d'une transaction brute."""
    dt = row["trans_datetime"]
    dob = row["dob"]

    hour        = dt.hour
    dow         = dt.weekday()
    month       = dt.month
    is_weekend  = int(dow >= 5)
    is_night    = int(hour >= 22 or hour <= 5)
    age         = int((dt.date() - dob).days // 365)

    # Distance Haversine
    R = 6371
    lat1, lat2 = np.radians(row["lat"]),  np.radians(row["merch_lat"])
    lon1, lon2 = np.radians(row["long"]), np.radians(row["merch_long"])
    a = (np.sin((lat2-lat1)/2)**2
         + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)
    dist = R * 2 * np.arcsin(np.sqrt(a))

    log_amt = np.log1p(row["amt"])

    # Encodage catégoriel
    features = {
        "amt": row["amt"],
        "log_amt": log_amt,
        "lat": row["lat"],
        "long": row["long"],
        "city_pop": row["city_pop"],
        "merch_lat": row["merch_lat"],
        "merch_long": row["merch_long"],
        "distance_km": dist,
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "age": age,
    }

    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        val = row[col]
        if val in le.classes_:
            features[col + "_enc"] = le.transform([val])[0]
        else:
            features[col + "_enc"] = 0  # valeur inconnue → 0

    return pd.DataFrame([features])


def predict_transaction(row, model, scaler, encoders, feature_cols):
    df_feat = compute_features(row, encoders)
    df_feat = df_feat[feature_cols].fillna(0)
    X_sc    = scaler.transform(df_feat)
    proba   = model.predict_proba(X_sc)[0][1]
    pred    = int(proba >= 0.5)
    return pred, proba


# ─────────────────────────────────────────────
# Chargement du dataset pour le dashboard
# ─────────────────────────────────────────────

@st.cache_data
def load_dataset(path="credit_card_fraud.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"]       = df["trans_date_trans_time"].dt.month
    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 Fraud Detection")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Dashboard", " Prédiction Manuelle", " Analyse par Lot", "ℹ️ À propos"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    # Statut modèle
    model, scaler, encoders, feature_cols, model_name = load_model_artifacts()
    if model is not None:
        st.success(f"Modèle chargé\n\n`{model_name}`")
    else:
        st.error("Modèle non trouvé\n\nLancez d'abord :\n```\npython train_model.py\n```")


# ─────────────────────────────────────────────
# PAGE : DASHBOARD
# ─────────────────────────────────────────────

if "Dashboard" in page:
    st.markdown('<p class="main-header">💳 Tableau de Bord — Détection de Fraude</p>', unsafe_allow_html=True)
    st.caption("Analyse exploratoire du dataset de transactions bancaires")

    df = load_dataset()

    if df is None:
        st.warning("Fichier `credit_card_fraud.csv` introuvable dans le répertoire courant.")
        st.info("Placez le fichier CSV dans le même dossier que `app.py` pour afficher les statistiques.")
    else:
        # ── KPIs ──
        total     = len(df)
        frauds    = df["is_fraud"].sum()
        legits    = total - frauds
        fraud_pct = frauds / total * 100
        avg_fraud_amt = df[df["is_fraud"]==1]["amt"].mean()
        avg_legit_amt = df[df["is_fraud"]==0]["amt"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions totales", f"{total:,}")
        c2.metric("Fraudes détectées", f"{frauds:,}", f"{fraud_pct:.2f}%")
        c3.metric("Montant moyen (fraude)", f"${avg_fraud_amt:.0f}")
        c4.metric("Montant moyen (légit.)", f"${avg_legit_amt:.0f}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Distribution des montants
            fig = px.histogram(
                df, x="amt", color=df["is_fraud"].map({0:"Légitime", 1:"Fraude"}),
                nbins=60, barmode="overlay", log_y=True,
                color_discrete_map={"Légitime":"#457b9d","Fraude":"#e63946"},
                title="Distribution des Montants",
                labels={"amt":"Montant ($)", "color":"Type"}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Fraudes par heure
            fraud_hour = df[df["is_fraud"]==1]["hour"].value_counts().sort_index()
            fig2 = px.bar(
                x=fraud_hour.index, y=fraud_hour.values,
                labels={"x":"Heure","y":"Nb fraudes"},
                title="Fraudes par Heure de la Journée",
                color=fraud_hour.values,
                color_continuous_scale="Reds"
            )
            fig2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            # Top catégories fraudées
            fraud_cat = (
                df[df["is_fraud"]==1]["category"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            fraud_cat.columns = ["Catégorie", "Nb fraudes"]
            fig3 = px.bar(
                fraud_cat, x="Nb fraudes", y="Catégorie",
                orientation="h", title="Top 10 Catégories (Fraudes)",
                color="Nb fraudes", color_continuous_scale="OrRd"
            )
            fig3.update_layout(height=380)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            # Taux de fraude par jour de semaine
            dow_map = {0:"Lun",1:"Mar",2:"Mer",3:"Jeu",4:"Ven",5:"Sam",6:"Dim"}
            df["dow_label"] = df["day_of_week"].map(dow_map)
            rate = df.groupby("day_of_week").agg(
                taux=("is_fraud","mean"), dow=("dow_label","first")
            ).reset_index()
            fig4 = px.line(
                rate, x="dow", y="taux",
                markers=True, title="Taux de Fraude par Jour",
                labels={"dow":"Jour","taux":"Taux de fraude"}
            )
            fig4.update_traces(line_color="#e63946", marker_color="#e63946")
            fig4.update_layout(height=380, yaxis_tickformat=".2%")
            st.plotly_chart(fig4, use_container_width=True)

        # Carte (scatter geo)
        st.subheader("Géolocalisation des Fraudes")
        sample_fraud = df[df["is_fraud"]==1].sample(min(500, frauds))
        fig5 = px.scatter_geo(
            sample_fraud, lat="lat", lon="long",
            color_discrete_sequence=["red"],
            scope="usa", opacity=0.5,
            title="Localisation des transactions frauduleuses (échantillon 500)"
        )
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE : PRÉDICTION MANUELLE
# ─────────────────────────────────────────────

elif "Manuelle" in page:
    st.markdown('<p class="main-header">🔬 Prédiction Manuelle</p>', unsafe_allow_html=True)
    st.caption("Saisissez les détails d'une transaction pour obtenir une prédiction en temps réel.")

    if model is None:
        st.error("Modèle non disponible. Veuillez d'abord exécuter `python train_model.py`.")
        st.stop()

    with st.form("transaction_form"):
        st.subheader("Détails de la Transaction")

        col1, col2, col3 = st.columns(3)
        with col1:
            amt       = st.number_input("Montant ($)", min_value=0.01, value=150.0, step=0.01)
            category  = st.selectbox("Catégorie", CATEGORIES, index=0)
            state     = st.selectbox("État (US)", US_STATES, index=4)

        with col2:
            trans_date = st.date_input("Date de transaction", value=date.today())
            trans_time = st.time_input("Heure de transaction", value=time(14, 30))
            city_pop   = st.number_input("Population de la ville", min_value=100, value=50000)

        with col3:
            lat       = st.number_input("Latitude titulaire", value=37.77, format="%.4f")
            lon       = st.number_input("Longitude titulaire", value=-122.41, format="%.4f")
            merch_lat = st.number_input("Latitude marchand", value=37.80, format="%.4f")
            merch_lon = st.number_input("Longitude marchand", value=-122.42, format="%.4f")
            dob       = st.date_input("Date de naissance", value=date(1985, 6, 15))

        submitted = st.form_submit_button("Analyser la transaction", use_container_width=True)

    if submitted:
        trans_dt = datetime.combine(trans_date, trans_time)

        row = {
            "trans_datetime": trans_dt,
            "amt": amt,
            "category": category,
            "state": state,
            "city_pop": city_pop,
            "lat": lat,
            "long": lon,
            "merch_lat": merch_lat,
            "merch_long": merch_lon,
            "dob": dob,
        }

        with st.spinner("Analyse en cours..."):
            pred, proba = predict_transaction(row, model, scaler, encoders, feature_cols)

        st.markdown("---")
        st.subheader("Résultat de l'Analyse")

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if pred == 1:
                st.markdown('<p class="fraud-badge"> FRAUDE DÉTECTÉE</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="legit-badge"> TRANSACTION LÉGITIME</p>', unsafe_allow_html=True)
            st.metric("Probabilité de fraude", f"{proba*100:.1f}%")
            st.metric("Confiance", f"{abs(proba - 0.5)*200:.0f}%")

        with col_res2:
            # Jauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Score de Fraude (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e63946" if pred else "#2a9d8f"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Détails calculés
        with st.expander("📐 Détails des features calculées"):
            R = 6371
            lat1r, lat2r = np.radians(lat), np.radians(merch_lat)
            lon1r, lon2r = np.radians(lon), np.radians(merch_lon)
            a = np.sin((lat2r-lat1r)/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin((lon2r-lon1r)/2)**2
            dist = R * 2 * np.arcsin(np.sqrt(a))

            details = {
                "Heure": trans_dt.hour,
                "Jour semaine": ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"][trans_dt.weekday()],
                "Mois": trans_dt.month,
                "Weekend": "Oui" if trans_dt.weekday() >= 5 else "Non",
                "Nuit (22h-5h)": "Oui" if (trans_dt.hour >= 22 or trans_dt.hour <= 5) else "Non",
                "Âge titulaire": int((trans_date - dob).days // 365),
                "Distance (km)": f"{dist:.1f}",
                "Log-montant": f"{np.log1p(amt):.4f}",
            }
            st.table(pd.DataFrame(details.items(), columns=["Feature", "Valeur"]))


# ─────────────────────────────────────────────
# PAGE : ANALYSE PAR LOT
# ─────────────────────────────────────────────

elif "Lot" in page:
    st.markdown('<p class="main-header"> Analyse par Lot</p>', unsafe_allow_html=True)
    st.caption("Importez un fichier CSV de transactions pour les analyser en masse.")

    if model is None:
        st.error("Modèle non disponible. Veuillez d'abord exécuter `python train_model.py`.")
        st.stop()

    uploaded = st.file_uploader(
        "Chargez votre fichier CSV (même format que le dataset d'entraînement)",
        type=["csv"]
    )

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.info(f" Fichier chargé : **{len(df_up):,} transactions**")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button(" Lancer l'analyse", use_container_width=True):
            with st.spinner(f"Analyse de {len(df_up):,} transactions..."):
                # Feature engineering
                df_up = df_up.copy()
                df_up["trans_date_trans_time"] = pd.to_datetime(df_up["trans_date_trans_time"])
                df_up["dob"] = pd.to_datetime(df_up["dob"])
                df_up["hour"]        = df_up["trans_date_trans_time"].dt.hour
                df_up["day_of_week"] = df_up["trans_date_trans_time"].dt.dayofweek
                df_up["month"]       = df_up["trans_date_trans_time"].dt.month
                df_up["is_weekend"]  = (df_up["day_of_week"] >= 5).astype(int)
                df_up["is_night"]    = ((df_up["hour"] >= 22) | (df_up["hour"] <= 5)).astype(int)
                df_up["age"]         = (df_up["trans_date_trans_time"] - df_up["dob"]).dt.days // 365
                df_up["log_amt"]     = np.log1p(df_up["amt"])

                R = 6371
                lat1 = np.radians(df_up["lat"]); lat2 = np.radians(df_up["merch_lat"])
                lon1 = np.radians(df_up["long"]); lon2 = np.radians(df_up["merch_long"])
                a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
                df_up["distance_km"] = R * 2 * np.arcsin(np.sqrt(a))

                for col in CATEGORICAL_FEATURES:
                    le = encoders[col]
                    df_up[col+"_enc"] = df_up[col].apply(
                        lambda v: le.transform([v])[0] if v in le.classes_ else 0
                    )

                X = df_up[feature_cols].fillna(0)
                X_sc = scaler.transform(X)
                probas = model.predict_proba(X_sc)[:, 1]
                df_up["fraud_score"]    = probas
                df_up["fraud_predicted"] = (probas >= 0.5).astype(int)

            st.success("Analyse terminée !")
            n_fraud = df_up["fraud_predicted"].sum()
            st.metric("Fraudes détectées", f"{n_fraud:,}",
                      f"{n_fraud/len(df_up)*100:.2f}% du lot")

            # Tableau des fraudes
            fraud_rows = df_up[df_up["fraud_predicted"]==1].copy()
            fraud_rows = fraud_rows.sort_values("fraud_score", ascending=False)

            st.subheader(f" {len(fraud_rows)} Transactions Suspectes")
            cols_show = [c for c in ["trans_date_trans_time","merchant","category",
                                     "amt","state","fraud_score"] if c in fraud_rows.columns]
            st.dataframe(
                fraud_rows[cols_show].head(50).style.format({"fraud_score":"{:.2%}"}),
                use_container_width=True
            )

            # Distribution des scores
            fig = px.histogram(
                df_up, x="fraud_score", nbins=50,
                title="Distribution des Scores de Fraude",
                color_discrete_sequence=["#e63946"],
                labels={"fraud_score": "Score de fraude"}
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="black",
                          annotation_text="Seuil 0.5")
            st.plotly_chart(fig, use_container_width=True)

            # Export CSV
            csv_out = df_up.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇Télécharger les résultats (CSV)",
                data=csv_out,
                file_name="resultats_fraude.csv",
                mime="text/csv",
                use_container_width=True
            )


# ─────────────────────────────────────────────
# PAGE : À PROPOS
# ─────────────────────────────────────────────

elif "propos" in page:
    st.markdown('<p class="main-header">ℹ À propos</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 💳 Credit Card Fraud Detection

    Cette application détecte les transactions bancaires frauduleuses grâce au machine learning.

    ---

    ####  Dataset
    | Propriété | Valeur |
    |-----------|--------|
    | Transactions | 339 607 |
    | Fraudes | 1 782 (0.52%) |
    | Features brutes | 15 |
    | Features ingéniées | 16 |

    ---

    ####  Modèles
    | Modèle | Technique |
    |--------|-----------|
    | **XGBoost** (principal) | Gradient Boosting avec `scale_pos_weight` |
    | **Random Forest** (fallback) | Forêts aléatoires avec `class_weight=balanced` |

    ---

    ####  Pipeline
    1. **Feature Engineering** — heure, jour, distance Haversine, âge, log-montant
    2. **SMOTE** — sur-échantillonnage synthétique pour équilibrer les classes
    3. **StandardScaler** — normalisation des features numériques
    4. **Entraînement** — optimisation avec ROC-AUC & Precision-Recall

    ---

    ####  Lancement rapide
    ```bash
    # 1. Entraîner le modèle
    python train_model.py --data credit_card_fraud.csv --output model/

    # 2. Lancer l'interface
    streamlit run app.py
    ```

    ---

    ####  Dépendances
    ```
    streamlit pandas numpy scikit-learn joblib
    plotly matplotlib seaborn xgboost imbalanced-learn
    ```
    """)
