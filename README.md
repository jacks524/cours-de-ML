🎓 Plan de Cours Complet – Machine Learning

🕒 Durée totale : 13 semaines
🎯 Niveau : Débutant → Intermédiaire
📘 Prérequis : Python basique, notions de maths niveau lycée

🧮 Module 1 – NumPy (3 semaines)
📅 Semaine 1 – Bases de NumPy
🔹 Introduction et installation

Présentation de l’écosystème Python pour le ML

Installation (pip / conda)

Jupyter Notebook vs script Python

Différences entre ndarray et listes Python

Avantages et performance vectorielle

🔹 Constructeurs de base
np.array(), np.zeros(), np.ones(), np.arange(), np.linspace(), np.random.randn()


Notion de distribution normale

Manipulation de shape et reshape

📅 Semaine 2 – Indexing et Slicing
🔹 Indexation

Accès 1D / 2D / 3D

Indexation négative

Indexation multiple avec listes

Masques booléens et filtrage conditionnel

🔹 Fancy Indexing

Sélection avancée avec tableaux d’indices

Cas pratiques : feature engineering

Exercices : extraction de sous-ensembles complexes

📅 Semaine 3 – Opérations avancées
🔹 Broadcasting

Règles et cas d’usage

Applications en algèbre linéaire

🔹 Statistiques

Moyenne, écart-type, variance, min, max

Opérations par axe, percentiles, médianes

🔹 Algèbre linéaire
np.dot(), @, np.transpose(), np.linalg.det(), np.linalg.inv()


Valeurs propres et vecteurs propres

Cas pratique : simulation d’un dataset ML

🧾 Module 2 – Pandas (2 semaines)
📅 Semaine 1 – Fondations
🔹 Structures principales

Series et DataFrame

Création depuis diverses sources

Index hiérarchiques

🔹 Manipulation

Sélection, tri, ajout/suppression de colonnes

Gestion des valeurs manquantes (détection, imputation, suppression)

📅 Semaine 2 – Opérations avancées
🔹 Agrégations et regroupements

groupby() et agrégations multiples

Pivot tables et restructuration

🔹 Fusion et concaténation

merge(), join(), concaténation horizontale/verticale

Gestion des clés et index

🔹 Séries temporelles

Dates et heures (datetime)

Resampling, rolling windows, analyses temporelles

📊 Module 3 – Data Visualization (1 semaine)
🔹 Introduction pratique

matplotlib.pyplot : figures, axes, sous-graphiques

Graphiques fondamentaux + personnalisation

🔹 Seaborn

Visualisations statistiques

Intégration avec pandas.DataFrame

Thèmes et palettes

🔹 Intégration Pandas

Plotting direct depuis les DataFrames

Création rapide de dashboards exploratoires

🤖 Module 4 – Scikit-learn : Apprentissage Supervisé (4 semaines)
🔹 Régression

Linear Regression — fondement mathématique

Decision Trees — interprétabilité, overfitting

Random Forest — performance robuste

Gradient Boosting (XGBoost, LightGBM) — SOTA, hyperparameter tuning

🔹 Classification

Logistic Regression — probabilités et métriques

SVM — kernels, frontières complexes

Random Forest — vote majoritaire, robustesse

XGBoost — régularisation, early stopping

🧩 Module 5 – Apprentissage Non Supervisé (2 semaines)
🔹 Clustering

K-Means : algorithme, initialisation, segmentation

PCA : réduction de dimension, variance expliquée

DBSCAN : détection de clusters denses, gestion du bruit

🔹 Validation et Visualisation

Silhouette score, métriques de cohérence

Visualisation 2D/3D des clusters

⚙️ Module 6 – Pipeline Complet (1 semaine)
🔹 Workflow professionnel

Pipeline, ColumnTransformer

Gestion des variables catégorielles / numériques

Reproductibilité et structuration du projet

🔹 Feature Engineering

Création de features non linéaires

Sélection automatique de variables

🔹 Validation croisée

KFold, StratifiedKFold

GridSearchCV, RandomizedSearchCV

🔹 Hyperparameter Tuning

Méthodes systématiques

Optimisation bayésienne

Compromis biais-variance

🧠 Recommandations Scikit-learn
🔸 Priorisation des algorithmes
Type	Algorithmes	Points forts
Régression	Linear Regression, Decision Trees, Random Forest	Interprétabilité & performance
Classification	Logistic Regression, Random Forest, SVM	Robustesse & flexibilité
Clustering	K-Means, DBSCAN	Données réelles & segmentation
🔸 Modules essentiels
# Structure du cours scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline, ColumnTransformer

📈 Progression pédagogique
Semaine	Thème principal	Objectif
1–2	Concepts fondamentaux	NumPy & Pandas
3–4	Visualisation & Linear Models	Scikit-learn bases
5–6	Modèles avancés & non supervisé	Tree-based, clustering
7	Intégration complète	Projet final
🧩 Évaluation & Projets
🔹 Évaluations formatives

Quiz hebdomadaires

Exercices de code progressifs

Mini-projets par module

🔹 Projet final

Dataset réel (ex. Kaggle)

Pipeline ML complet : de la préparation à la présentation

Analyse, optimisation, insights et rapport

🎯 Compétences visées

✅ Maîtriser l’écosystème Python du Machine Learning
✅ Construire un pipeline complet d’apprentissage automatique
✅ Comprendre et comparer les modèles supervisés / non supervisés
✅ Effectuer du feature engineering efficace
✅ Évaluer et améliorer la performance des modèles
