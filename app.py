import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================
# CONFIGURATION PAGE
# ============================================
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="centered"
)

# ============================================
# STYLE CSS
# ============================================
st.markdown("""
<style>
    .main { background-color: #F8FAFF; }
    .stButton > button {
        background-color: #1A5FA8;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 40px;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0C3866;
    }
    .risk-high {
        background-color: #FFEBEE;
        border-left: 5px solid #C62828;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .risk-medium {
        background-color: #FFF3E0;
        border-left: 5px solid #E65100;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .risk-low {
        background-color: #EAF3DE;
        border-left: 5px solid #2E7D32;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #E6F1FB;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CHARGEMENT MODELE
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load('credit_risk_model.pkl')
    with open('model_columns.json', 'r') as f:
        columns = json.load(f)
    return model, columns

model, model_columns = load_model()

# ============================================
# HEADER
# ============================================
st.markdown("## 🏦 Credit Risk Predictor")
st.markdown("**Modèle XGBoost · AUC-ROC 0.7646 · Home Credit Default Risk**")
st.markdown("---")
st.markdown("Entrez les informations du client pour obtenir une prédiction de risque de défaut de paiement.")

# ============================================
# FORMULAIRE
# ============================================
st.markdown("### 👤 Profil du client")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Âge du client", min_value=20, max_value=70, value=35, step=1)
    
    gender = st.selectbox("Genre", ["Femme", "Homme"])
    
    education = st.selectbox("Niveau d'éducation", [
        "Higher education",
        "Secondary / secondary special",
        "Incomplete higher",
        "Lower secondary"
    ])

with col2:
    income = st.slider(
        "Revenu annuel (€)",
        min_value=20000, max_value=500000,
        value=120000, step=5000,
        format="%d €"
    )
    
    credit = st.slider(
        "Montant du crédit (€)",
        min_value=10000, max_value=1000000,
        value=200000, step=10000,
        format="%d €"
    )
    
    annuity = st.slider(
        "Mensualité (€)",
        min_value=500, max_value=50000,
        value=10000, step=500,
        format="%d €"
    )

st.markdown("### 📊 Scores et indicateurs")

col3, col4 = st.columns(2)

with col3:
    ext_source_2 = st.slider(
        "Score externe 2 (bureau de crédit)",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.01,
        help="Score fourni par un bureau de crédit externe. Plus élevé = meilleur profil."
    )
    
    ext_source_3 = st.slider(
        "Score externe 3 (bureau de crédit)",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.01,
        help="Score fourni par un second bureau de crédit. Plus élevé = meilleur profil."
    )

with col4:
    employment_years = st.slider(
        "Ancienneté emploi (années)",
        min_value=0, max_value=40,
        value=5, step=1
    )
    
    region_rating = st.selectbox(
        "Note de la région",
        [1, 2, 3],
        index=1,
        help="1 = région favorable, 3 = région à risque élevé"
    )

# ============================================
# PREDICTION
# ============================================
if st.button("🔍 Prédire le risque de défaut"):

    # Construire les features engineerées
    credit_income_ratio  = credit / income
    annuity_income_ratio = annuity / income
    credit_goods_ratio   = 1.1  # valeur par défaut
    loan_duration_years  = credit / annuity / 12
    ext_source_mean      = (ext_source_2 + ext_source_3) / 2
    ext_source_weighted  = ext_source_2 * 0.45 + ext_source_3 * 0.55
    ext_source_min       = min(ext_source_2, ext_source_3)
    ext_source_diff      = abs(ext_source_2 - ext_source_3)
    age_years            = float(age)
    employment_age_ratio = employment_years / age if age > 0 else 0

    # Construire le vecteur de features
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0  # initialiser à 0

    # Remplir les features disponibles
    feature_map = {
        'AGE_YEARS'            : age_years,
        'AMT_INCOME_TOTAL'     : float(income),
        'AMT_CREDIT'           : float(credit),
        'AMT_ANNUITY'          : float(annuity),
        'EXT_SOURCE_2'         : ext_source_2,
        'EXT_SOURCE_3'         : ext_source_3,
        'EXT_SOURCE_MEAN'      : ext_source_mean,
        'EXT_SOURCE_WEIGHTED'  : ext_source_weighted,
        'EXT_SOURCE_MIN'       : ext_source_min,
        'EXT_SOURCE_DIFF'      : ext_source_diff,
        'CREDIT_INCOME_RATIO'  : credit_income_ratio,
        'ANNUITY_INCOME_RATIO' : annuity_income_ratio,
        'CREDIT_GOODS_RATIO'   : credit_goods_ratio,
        'LOAN_DURATION_YEARS'  : loan_duration_years,
        'EMPLOYMENT_YEARS'     : float(employment_years),
        'EMPLOYMENT_AGE_RATIO' : employment_age_ratio,
        'REGION_RATING_CLIENT' : float(region_rating),
        'REGION_RATING_CLIENT_W_CITY': float(region_rating),
    }

    for feat, val in feature_map.items():
        if feat in input_data.columns:
            input_data.loc[0, feat] = val

    # Encodage genre
    if 'CODE_GENDER_M' in input_data.columns:
        input_data.loc[0, 'CODE_GENDER_M'] = 1 if gender == "Homme" else 0

    # Encodage éducation
    edu_col = f'NAME_EDUCATION_TYPE_{education}'
    if edu_col in input_data.columns:
        input_data.loc[0, edu_col] = 1

    # Convertir en float
    input_data = input_data.astype(float)

    # Prédiction
    risk_score = model.predict_proba(input_data)[0][1]
    risk_pct   = risk_score * 100

    # ── Résultat ────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Résultat de l'analyse")

    if risk_score >= 0.6:
        risk_label = "⚠️ RISQUE ÉLEVÉ"
        risk_class = "risk-high"
        color       = "#C62828"
        decision    = "❌ Demander des garanties supplémentaires"
        advice      = "Ce profil présente un risque de défaut élevé. Recommandation : exiger des garanties additionnelles ou réduire le montant du crédit."
    elif risk_score >= 0.3:
        risk_label = "⚡ RISQUE MODÉRÉ"
        risk_class = "risk-medium"
        color       = "#E65100"
        decision    = "⚠️ Analyse approfondie recommandée"
        advice      = "Ce profil présente un risque modéré. Recommandation : analyser l'historique de crédit complet avant décision."
    else:
        risk_label = "✅ RISQUE FAIBLE"
        risk_class = "risk-low"
        color       = "#2E7D32"
        decision    = "✅ Approbation accélérée possible"
        advice      = "Ce profil présente un faible risque de défaut. Recommandation : processus d'approbation accéléré."

    st.markdown(f"""
    <div class="{risk_class}">
        <h2 style="color:{color}; margin:0">{risk_label}</h2>
        <h1 style="color:{color}; margin:5px 0">{risk_pct:.1f}%</h1>
        <p style="font-size:16px; margin:5px 0">Probabilité de défaut de paiement</p>
        <p style="font-size:18px; font-weight:bold; margin:10px 0">{decision}</p>
        <p style="color:#555; margin:0">{advice}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Métriques clés ────────────────────────
    st.markdown("### 📊 Indicateurs clés du profil")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        ratio_str = f"{credit_income_ratio:.1f}x"
        color_r = "#C62828" if credit_income_ratio > 5 else "#2E7D32"
        st.markdown(f"""
        <div class="metric-card">
            <p style="margin:0;font-size:12px;color:#666">Crédit / Revenu</p>
            <p style="margin:0;font-size:22px;font-weight:bold;color:{color_r}">{ratio_str}</p>
        </div>""", unsafe_allow_html=True)

    with m2:
        dur_str = f"{loan_duration_years:.1f} ans"
        color_d = "#C62828" if loan_duration_years > 3 else "#2E7D32"
        st.markdown(f"""
        <div class="metric-card">
            <p style="margin:0;font-size:12px;color:#666">Durée prêt</p>
            <p style="margin:0;font-size:22px;font-weight:bold;color:{color_d}">{dur_str}</p>
        </div>""", unsafe_allow_html=True)

    with m3:
        ext_str = f"{ext_source_mean:.2f}"
        color_e = "#2E7D32" if ext_source_mean > 0.5 else "#C62828"
        st.markdown(f"""
        <div class="metric-card">
            <p style="margin:0;font-size:12px;color:#666">Score Ext. Moyen</p>
            <p style="margin:0;font-size:22px;font-weight:bold;color:{color_e}">{ext_str}</p>
        </div>""", unsafe_allow_html=True)

    with m4:
        ann_str = f"{annuity_income_ratio*100:.1f}%"
        color_a = "#C62828" if annuity_income_ratio > 0.3 else "#2E7D32"
        st.markdown(f"""
        <div class="metric-card">
            <p style="margin:0;font-size:12px;color:#666">Charge mensuelle</p>
            <p style="margin:0;font-size:22px;font-weight:bold;color:{color_a}">{ann_str}</p>
        </div>""", unsafe_allow_html=True)

    # ── SHAP Waterfall ────────────────────────
    st.markdown("### 🔍 Pourquoi ce score ? (Explication SHAP)")
    st.caption("Les barres rouges augmentent le risque · Les barres bleues le réduisent")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        top_features = pd.DataFrame({
            'Feature' : input_data.columns,
            'SHAP'    : shap_values[0]
        })
        top_features = top_features[top_features['SHAP'] != 0]
        top_features = top_features.reindex(
            top_features['SHAP'].abs().sort_values(ascending=False).index
        ).head(8)

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ['#C62828' if v > 0 else '#1A5FA8' for v in top_features['SHAP']]
        bars = ax.barh(top_features['Feature'], top_features['SHAP'], color=colors)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Impact sur le score de risque', fontsize=11)
        ax.set_title('Facteurs influençant la prédiction', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)

        for bar, val in zip(bars, top_features['SHAP']):
            ax.text(
                val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}',
                va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.info("Explication SHAP non disponible pour cette configuration.")

    # ── Footer ────────────────────────────────
    st.markdown("---")
    st.caption(
        "Modèle XGBoost entraîné sur 246 008 clients · "
        "Dataset : Home Credit Default Risk (Kaggle) · "
        "AUC-ROC : 0.7646 · Recall défaut : 65.98%"
    )

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 🏦 Credit Risk Predictor")
    st.markdown("**Projet Portfolio — Lynda**")
    st.markdown("---")
    st.markdown("**Modèle :** XGBoost")
    st.markdown("**AUC-ROC :** 0.7646")
    st.markdown("**Recall défaut :** 65.98%")
    st.markdown("**Dataset :** 307 511 clients")
    st.markdown("---")
    st.markdown("**Top drivers du défaut :**")
    st.markdown("1. Score externe composite")
    st.markdown("2. Durée du prêt")
    st.markdown("3. Ratio crédit/revenu")
    st.markdown("4. Âge du client")
    st.markdown("5. Genre")
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/tawhida-lab/credit-risk-prediction)")
