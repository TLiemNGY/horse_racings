# Projet de Prédiction et Stratégies d'Investissement

## Description
Ce projet utilise des modèles de Machine Learning (XGBoost, RandomForest, etc.) pour prédire des tendances à partir de données de courses et tester différentes stratégies d'investissement.

## Structure du Projet
```
Nom_du_projet/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── interim/
│   ├── runs.csv
│   ├── races.csv
│
├── notebooks/
│   ├── EDA.ipynb
│
├── src/
│   ├── data_processing/
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   ├── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │
│   ├── investment_strategies/
│       ├── strategy_1.py
│       ├── strategy_2.py
│
├── tests/
│
├── config/
│   ├── config.yaml
│
├── .gitignore
├── requirements.txt
├── setup.py
├── README.md
```

## Installation
1. Assurez-vous d'avoir Python 3.11 installé sur votre système.
2. Si vous n'avez pas encore d'environnement virtuel pour ce projet, créez-en un avec :
   ```bash
   py -3.11 -m venv .venv
   ```
3. Activez l'environnement virtuel :
   - Sous macOS/Linux :
     ```bash
     source venv/bin/activate
     ```
   - Sous Windows :
     ```bash
     source .venv/Scripts/activate
     ```
4. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
```bash
python src/models/train.py
```

