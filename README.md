Plan de Cours Complet - Machine Learning
Module 1 : NumPy (3 semaines)
Semaine 1 : Bases de NumPy
Introduction et installation

Présentation de l'écosystème Python pour le ML

Installation des bibliothèques (pip/conda)

Jupyter Notebook vs scripts Python

ndarray vs listes Python

Comparaison performance : opérations vectorielles

Types de données et optimisation mémoire

Avantages pour les calculs scientifiques

Constructeurs de base

np.array(), np.zeros(), np.ones()

np.arange(), np.linspace()

np.random.randn() et distribution normale

Manipulation shape/reshape

Attribut shape et dimensionnalité

Méthode reshape() et règles de transformation

Concepts de lignes, colonnes et axes

Semaine 2 : Indexing et Slicing
Indexing basique et avancé

Accès aux éléments (1D, 2D, 3D)

Indexation négative et bornes

Indexing multiple avec listes

Masques booléens

Création de conditions sur les tableaux

Filtrage de données avec conditions multiples

Applications en prétraitement

Fancy indexing

Sélection avancée avec tableaux d'indices

Combinaison d'indexing et slicing

Cas d'usage en feature engineering

Exercices pratiques

Manipulation de datasets synthétiques

Extraction de sous-ensembles complexes

Applications sur données réelles

Semaine 3 : Opérations avancées
Broadcasting

Règles du broadcasting

Opérations entre tableaux de tailles différentes

Applications en algèbre linéaire

Opérations statistiques

mean(), std(), var(), min(), max()

Opérations par axe (lignes/colonnes)

Percentiles et médianes

Algèbre linéaire

Produit matriciel (np.dot(), @)

Transposition, déterminant, inverse

Valeurs propres et vecteurs propres

Cas pratique complet

Simulation de dataset ML complet

Prétraitement et feature engineering

Préparation pour modèles ML

Module 2 : Pandas (2 semaines)
Semaine 1 : Fondations
Series et DataFrames

Création à partir de diverses sources

Structure et propriétés fondamentales

Index personnalisés et hiérarchiques

Manipulation de données tabulaires

Sélection de colonnes et lignes

Ajout/suppression de colonnes

Tri et organisation des données

Gestion des missing values

Détection des valeurs manquantes

Stratégies d'imputation

Suppression sélective

Semaine 2 : Opérations avancées
Aggrégations et groupby

Opérations de regroupement complexes

Agrégations multiples et personnalisées

Pivot tables et restructuration

Fusion de datasets

Jointures SQL-like (merge, join)

Concatenation verticale/horizontale

Gestion des clés et index

Manipulation de time series

Dates et heures avec datetime

Resampling et rolling windows

Analyses temporelles

Module 3 : Data Visualization (1 semaine)
Introduction pratique
matplotlib.pyplot (basics)

Figures, axes et sous-graphiques

Types de graphiques fondamentaux

Personnalisation (couleurs, styles, labels)

Seaborn pour visualisations statistiques

Graphiques statistiques avancés

Intégration avec les DataFrames pandas

Thèmes et styles prédéfinis

Intégration avec Pandas

Méthodes de plotting des DataFrames

Visualisation directe depuis les données

Dashboard rapide pour exploration

Module 4 : Scikit-learn - Apprentissage Supervisé (4 semaines)
Régression
1. Linear Regression (fondamental)
Concepts des moindres carrés

Interprétation des coefficients

Évaluation des performances

2. Decision Trees (explicable)
Arbres de décision pour régression

Importance des features

Limites et sur-apprentissage

3. Random Forest (performant)
Principe du bagging et forêts aléatoires

Réduction de la variance

Feature importance globale

4. Gradient Boosting (state-of-the-art)
Boosting séquentiel

XGBoost, LightGBM applications

Hyperparameter tuning avancé

Classification
1. Logistic Regression (base)
Régression logistique binaire et multiclasse

Probabilités et seuils de décision

Métriques de classification

2. SVM (frontières complexes)
Séparateurs à marge maximale

Kernels pour non-linéarité

Cas d'usage en haute dimension

3. Random Forest (robuste)
Adaptation pour la classification

Vote majoritaire et probabilités

Robustesse au bruit

4. XGBoost (hautes performances)
Optimisation gradient boosting

Régularisation et early stopping

Compétitions et applications réelles

Module 5 : Apprentissage Non-Supervisé (2 semaines)
Clustering
K-Means Clustering

Algorithme et initialisation

Choix du nombre de clusters

Applications en segmentation

PCA - Réduction de dimension

Composantes principales

Variance expliquée

Visualisation en basse dimension

DBSCAN - Clustering density-based

Détection de clusters non-sphériques

Gestion du bruit

Paramétrisation avancée

Validation des clusters

Métriques de qualité (silhouette, etc.)

Interprétation des résultats

Visualisation des clusters

Module 6 : Pipeline Complet (1 semaine)
Workflow professionnel
Preprocessing avec sklearn

Pipelines et ColumnTransformers

Gestion des types de variables

Reproductibilité

Feature engineering

Création de features interactives

Transformations non-linéaires

Sélection automatique de features

Validation croisée

K-fold et stratified sampling

GridSearch et RandomSearch

Évaluation robuste des modèles

Hyperparameter tuning

Méthodologies systématiques

Compromis biais-variance

Optimisation bayésienne

Recommandations pour scikit-learn
Priorisation des algorithmes
Régression :
Linear Regression → Fondamentaux mathématiques

Decision Trees → Interprétabilité

Random Forest → Performance robuste

Classification :
Logistic Regression → Base probabiliste

Random Forest → Robustesse générale

SVM → Frontières complexes

Clustering :
K-Means → Standard industriel

DBSCAN → Données réelles complexes

Modules sklearn essentiels
python
# Structure du cours scikit-learn
- `sklearn.model_selection` 
  → train_test_split, cross_val_score, GridSearchCV

- `sklearn.preprocessing`
  → StandardScaler, LabelEncoder, OneHotEncoder

- `sklearn.metrics`
  → accuracy_score, precision_recall_fscore_support, mean_squared_error

- `sklearn.ensemble`
  → RandomForestClassifier, GradientBoostingRegressor

- `sklearn.pipeline`
  → Pipeline, ColumnTransformer, make_pipeline
Progression pédagogique recommandée
Semaines 1-2 : Concepts fondamentaux + Linear Models

Semaines 3-4 : Tree-based models + Optimisation

Semaines 5-6 : Non-supervisé + Cas pratiques

Semaine 7 : Intégration et projets

Évaluation et Projets
Évaluations formatives
Quiz hebdomadaires sur les concepts

Exercices de coding progressifs

Mini-projets par module

Projet final
Dataset réel de compétition Kaggle

Pipeline complet de A à Z

Présentation des résultats et insights

Compétences visées
Maîtrise de l'écosystème Python ML

Capacité à résoudre des problèmes business

Compétences en feature engineering

Évaluation rigoureuse des modèles

Durée totale estimée : 13 semaines
Niveau : Débutant à Intermédiaire
Prérequis : Python basique, mathématiques lycée

