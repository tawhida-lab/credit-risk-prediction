# Credit Risk Prediction 🏦

## Contexte
Prédiction de la probabilité de défaut de paiement  
307 511 clients · 122 variables · Kaggle Competition

## Stack technique
Python · XGBoost · SHAP · scikit-learn · SMOTE · Pandas · Seaborn

## Résultats
| Modèle | AUC-ROC | Recall Défaut |
|--------|---------|---------------|
| Logistic Regression | 0.7209 | 2.22% |
| Random Forest | 0.6916 | 26.22% |
| **XGBoost ✓** | **0.7646** | **65.98%** |

## Structure du projet
- Phase 1 : EDA & Nettoyage (122 → 75 colonnes)
- Phase 2 : Feature Engineering (11 nouvelles variables)
- Phase 3 : Modélisation ML (3 modèles comparés)
- Phase 4 : Interprétabilité SHAP
- Phase 5 : Recommandations business

## Top 5 drivers du défaut (SHAP)
1. EXT_SOURCE_WEIGHTED (score externe composite)
2. EXT_SOURCE_MEAN (moyenne scores externes)
3. LOAN_DURATION_YEARS (durée du prêt)
4. CODE_GENDER_M (genre)
5. CREDIT_GOODS_RATIO (ratio crédit/valeur bien)

## Dataset
[Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
