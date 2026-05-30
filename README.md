# Credit Risk Prediction 🏦

## Contexte
Prédiction de la probabilité de défaut de paiement  
307 511 clients · 122 variables · 8 tables · Kaggle Competition

## Stack technique
Python · XGBoost · SHAP · scikit-learn · SMOTE · Pandas · Seaborn · Power BI

## Résultats
| Modèle | AUC-ROC | Recall Défaut |
|--------|---------|---------------|
| Logistic Regression | 0.7209 | 2.22% |
| Random Forest | 0.6916 | 26.22% |
| **XGBoost ✓** | **0.7646** | **65.98%** |

## Structure du projet
- Phase 1 : EDA & Nettoyage (122 → 75 colonnes)
- Phase 2 : Feature Engineering (11 nouvelles variables créées)
- Phase 3 : Modélisation ML (3 modèles comparés)
- Phase 4 : Interprétabilité SHAP
- Phase 5 : Recommandations business
- Phase 6 : Dashboard Power BI (Star Schema — 3 tables)

## Top 5 drivers du défaut (SHAP)
1. EXT_SOURCE_WEIGHTED (score externe composite — créé)
2. EXT_SOURCE_MEAN (moyenne scores externes — créé)
3. LOAN_DURATION_YEARS (durée du prêt — créé)
4. CODE_GENDER_M (genre masculin)
5. CREDIT_GOODS_RATIO (ratio crédit/valeur bien — créé)

## Dashboard Power BI
4 pages interactives construites sur les vrais résultats du modèle XGBoost :
- Vue Executive — KPIs globaux + distribution du risque
- Profils de Risque — analyse par âge et genre
- Performance Modèles — comparaison AUC/Recall/F1
- SHAP Feature Importance — top 10 drivers du défaut

Modélisation des données : Star Schema
- `Fact_Clients` — 61 503 clients avec scores de risque
- `Dim_Modeles` — comparaison des 3 modèles
- `Dim_SHAP` — top 10 features SHAP

## Dataset
[Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
