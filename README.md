# README.md
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
```bash
pip install -r requirements.txt
```

## Utilisation
```bash
python src/models/train.py
```

---
