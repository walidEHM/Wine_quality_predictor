# Prédicteur de Qualité du Vin 🍷

## Description

Cette application web est un prédicteur de qualité du vin rouge basé sur l'apprentissage automatique. Elle utilise un modèle de forêt aléatoire entraîné sur un dataset de vins rouges pour estimer la qualité d'un vin en fonction de ses caractéristiques chimiques.

L'application est développée avec Flask et propose une interface web simple pour saisir les valeurs des 11 caractéristiques du vin et obtenir une prédiction de qualité sur une échelle de 3 à 8.

## Fonctionnalités

- **Prédiction en temps réel** : Saisissez les valeurs des caractéristiques chimiques et obtenez une prédiction instantanée.
- **Probabilités détaillées** : Visualisez les probabilités pour chaque niveau de qualité possible.
- **Interface utilisateur intuitive** : Formulaire web élégant avec validation des entrées.
- **API REST** : Endpoint `/predict` pour des intégrations programmatiques.

## Caractéristiques utilisées

Le modèle prend en compte 11 caractéristiques chimiques du vin :

1. Fixed acidity (acidité fixe)
2. Volatile acidity (acidité volatile)
3. Citric acid (acide citrique)
4. Residual sugar (sucre résiduel)
5. Chlorides (chlorures)
6. Free sulfur dioxide (dioxyde de soufre libre)
7. Total sulfur dioxide (dioxyde de soufre total)
8. Density (densité)
9. pH
10. Sulphates (sulfates)
11. Alcohol (alcool)

## Installation

### Prérequis

- Python 3.8
- Pip pour la gestion des paquets

### Étapes d'installation

1. Clonez ce dépôt ou téléchargez les fichiers dans un dossier.

2. Installez les dépendances :
   ```
   pip install flask joblib pandas scikit-learn
   ```

3. Assurez-vous que les fichiers du modèle sont présents :
   - `random_forest_model.pkl` : Le modèle entraîné
   - `scaler.pkl` : Le scaler pour la normalisation des données

4. Lancez l'application :
   ```
   python app.py
   ```

5. Ouvrez votre navigateur et allez à `http://localhost:5000`.

## Utilisation

1. Sur la page d'accueil, remplissez le formulaire avec les valeurs des 11 caractéristiques du vin.
2. Cliquez sur "Prédire la qualité" pour obtenir le résultat.
3. La prédiction inclut le niveau de qualité estimé et les probabilités pour chaque classe.

### API

Pour une utilisation programmatique, envoyez une requête POST à `/predict` avec les données JSON contenant les 11 caractéristiques.

Exemple avec curl :
```
curl -X POST http://localhost:5000/predict \
  -d "fixed acidity=7.4&volatile acidity=0.7&citric acid=0&residual sugar=1.9&chlorides=0.076&free sulfur dioxide=11&total sulfur dioxide=34&density=0.9978&pH=3.51&sulphates=0.56&alcohol=9.4"
```

## Technologies utilisées

- **Flask** : Framework web Python
- **Scikit-learn** : Bibliothèque d'apprentissage automatique
- **Joblib** : Sérialisation des modèles
- **Pandas** : Manipulation des données
- **HTML/CSS** : Interface utilisateur

## Structure du projet

```
app/
├── app.py                 # Application Flask principale
├── readme.md              # Ce fichier
├── templates/
│   └── index.html         # Template de la page d'accueil
├── random_forest_model.pkl # Modèle entraîné (à fournir)
└── scaler.pkl             # Scaler (à fournir)
```
