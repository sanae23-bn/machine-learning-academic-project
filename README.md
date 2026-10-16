# Prédiction du Churn des Joueurs en Ligne

Comparaison de la **Régression Logistique**, du **Random Forest** et de **XGBoost** avec rééchantillonnage **SMOTE**.

Projet réalisé dans le cadre du module *Machine Learning* du **Master IADAIG** (2025/2026), ENS Fès, Université Sidi Mohamed Ben Abdellah.

## Problématique

Comment construire un pipeline de classification supervisée permettant de prédire le churn des joueurs en ligne, en traitant le déséquilibre des classes et en maximisant la détection des joueurs à risque d'abandon ?

## Dataset

- **Predict Online Gaming Behavior Dataset** (Rabie El Kharoua, Kaggle, licence CC BY 4.0)
- 40 034 observations, 13 variables
- Cible : `EngagementLevel` recodée en binaire (`Low` = churn = 1, sinon 0), soit environ 25,8 % de churners
- Lien : https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset

> Le dataset n'est pas inclus dans ce dépôt : téléchargez-le depuis Kaggle.

## Pipeline

1. EDA et contrôle qualité des données
2. Encodage des variables catégorielles (`LabelEncoder`)
3. Feature engineering : `WeeklyPlayHours`, `AchievementPerLevel`, `EngagementScore`
4. Split train/test 80/20 stratifié
5. Normalisation (`StandardScaler`, ajusté sur le train uniquement)
6. SMOTE (sur le train uniquement)
7. Entraînement de 3 modèles et validation croisée stratifiée 5-fold
8. Évaluation : Accuracy, F1, Précision, Recall, AUC-ROC

## Résultats (jeu de test)

| Modèle | Accuracy | F1 | Précision | Recall | AUC-ROC |
|---|---|---|---|---|---|
| Régression Logistique | 0,87 | 0,78 | 0,70 | 0,89 | 0,87 |
| Random Forest | 0,94 | 0,89 | 0,90 | 0,88 | 0,93 |
| **XGBoost** | **0,95** | **0,91** | **0,92** | 0,88 | **0,94** |

**Variables les plus prédictives :** `SessionsPerWeek` (r = -0,50), `AvgSessionDurationMinutes` (r = -0,34) et la feature construite `EngagementScore`.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `projet_churn_IADAIG.ipynb` | Notebook complet (EDA, preprocessing, modélisation, évaluation) |
| `rpprt_churn.pdf` | Rapport scientifique du projet |

## Exécution

Le notebook a été développé sous Google Colab.

```bash
pip install -r requirements.txt
jupyter notebook projet_churn_IADAIG.ipynb
```

## Limites et perspectives

- Données synthétiques (distributions très propres, performances probablement optimistes)
- Pas de dimension temporelle (données en coupe transversale)
- Hyperparamètres fixés manuellement
- Perspectives : optimisation bayésienne (Optuna), explicabilité SHAP, seuil de décision optimisé (F2), modèles temporels et de survie, déploiement via API

## Équipe

- Sanae BENTALEB
- Bilal BAJOU
- Abdelillah ELATTARI
- Othmane AMRAOUY

**Encadrant :** Pr. Amine SALLAH
