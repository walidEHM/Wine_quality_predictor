# Prédicteur de Qualité du Vin Rouge 🍷

## Description

Ce projet de machine learning vise à prédire la qualité du vin rouge à partir de différentes caractéristiques physico-chimiques.

Les notebooks `Classification_red_wine.ipynb` et `forest.ipynb` présentent l'ensemble du pipeline de data science :
- exploration des données
- prétraitement
- visualisation
- entraînement des modèles
- optimisation des hyperparamètres
- sauvegarde du modèle final

Le modèle retenu est un **Voting Classifier** combinant :
- K-Nearest Neighbors (KNN)
- Random Forest

L'optimisation est réalisée avec **GridSearchCV** afin d'obtenir les meilleures performances possibles.

---

## Technologies et Bibliothèques Utilisées

Les bibliothèques nécessaires à l'exécution du notebook sont :
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- joblib

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

---

## Dataset

Le projet utilise le dataset `winequality-red.csv`, contenant plusieurs propriétés chimiques du vin rouge.

### Variables du dataset

| Variable | Description |
|----------|-------------|
| fixed acidity | Acidité fixe |
| volatile acidity | Acidité volatile |
| citric acid | Acide citrique |
| residual sugar | Sucre résiduel |
| chlorides | Chlorures |
| free sulfur dioxide | Dioxyde de soufre libre |
| total sulfur dioxide | Dioxyde de soufre total |
| density | Densité |
| pH | pH |
| sulphates | Sulfates |
| alcohol | Alcool |
| quality | **Variable cible** |

La variable cible `quality` représente une note comprise entre 3 et 8.

---

## Pipeline de Prétraitement

Le notebook applique plusieurs étapes importantes de préparation des données.

### 1. Chargement et Exploration

- Lecture du dataset avec pandas
- Analyse des dimensions et types de données
- Vérification des valeurs manquantes et doublons
- Statistiques descriptives

### 2. Visualisation des Données

- Pairplots
- Matrice de corrélation
- Boxplots pour l'analyse des outliers

### 3. Traitement des Valeurs Extrêmes

Les outliers sont réduits grâce à la méthode **IQR (Interquartile Range)**.

### 4. Normalisation

Les variables sont normalisées avec **MinMaxScaler** afin d'améliorer les performances des modèles.

### 5. Transformation de la Variable Cible

Les notes de qualité sont regroupées en 3 catégories :

| Catégorie | Description |
|-----------|-------------|
| 0 | Mauvaise qualité |
| 1 | Qualité moyenne |
| 2 | Bonne qualité |

### 6. Gestion du Déséquilibre des Classes

Le dataset étant déséquilibré, la technique **SMOTE** est utilisée pour équilibrer les classes.

---

## Modèles Entraînés

### 🔹 K-Nearest Neighbors (KNN)

Premier modèle utilisé comme baseline.

### 🔹 Random Forest

- Sélection des meilleures features avec **SelectKBest**
- Optimisation via **GridSearchCV**

### 🔹 Voting Classifier (Modèle Final)

Combinaison de :
- KNN
- Random Forest

Le **soft voting** est utilisé pour améliorer les performances globales du système.

---

## Performance du Modèle

Le modèle final atteint une précision d'environ **X%** sur les données de validation.

*Remplacez cette valeur par votre score réel obtenu dans le notebook.*

Les métriques utilisées incluent :
- Accuracy
- Classification Report
- Confusion Matrix

---

## Sauvegarde du Modèle

Le modèle entraîné ainsi que le scaler sont sauvegardés au format pickle dans le dossier `app/` :
- `app/random_forest_model.pkl`
- `app/scaler.pkl`

---

## Utilisation du Modèle

Exemple d'utilisation du modèle sur de nouvelles données :

```python
import pickle
import pandas as pd

# Chargement du modèle et du scaler
with open('app/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('app/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Exemple de nouvelles données
new_data = pd.DataFrame({
    'fixed acidity': [7.4],
    'volatile acidity': [0.7],
    'citric acid': [0.0],
    'residual sugar': [1.9],
    'chlorides': [0.076],
    'free sulfur dioxide': [11.0],
    'total sulfur dioxide': [34.0],
    'density': [0.9978],
    'pH': [3.51],
    'sulphates': [0.56],
    'alcohol': [9.4]
})

# Normalisation
new_data_scaled = scaler.transform(new_data)

# Prédiction
prediction = model.predict(new_data_scaled)
print(prediction)
```

### Interprétation des Résultats

| Valeur | Signification |
|--------|---------------|
| 0 | Mauvaise qualité |
| 1 | Qualité moyenne |
| 2 | Bonne qualité |

---

## Structure du Projet

```
├── Classification_red_wine.ipynb
├── forest.ipynb
├── readme.md
├── app/
│   ├── app.py
│   ├── readme.md
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── templates/
│       └── index.html
└── datasets/
    ├── winequality-red.csv
    ├── winequality-white.csv
    └── winequality.names
```

---

## Perspectives d'Amélioration

Quelques améliorations possibles :
- Tester d'autres algorithmes (XGBoost, LightGBM, CatBoost)
- Déployer le modèle avec Flask ou FastAPI
- Créer une interface web interactive
- Ajouter une API de prédiction
- Effectuer une optimisation avancée des hyperparamètres

---

## Auteur

Projet réalisé dans le cadre du Club AIOIT (IFRI).
