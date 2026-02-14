# 📊 Churn Prediction Web Application

Une application web moderne de **prédiction du churn client** basée sur l'IA, construite avec FastAPI et un modèle de machine learning KNN (K-Nearest Neighbors).

## 🎯 Objectif

Prédire si un client d'une entreprise de télécommunications va quitter le service ou rester. L'application offre deux modes :
- **Prédiction individuelle** : Analyser un client unique
- **Prédiction en batch** : Traiter plusieurs clients via upload CSV

## 🚀 Fonctionnalités

✅ **Interface intuitive** - Application web responsive avec formulaires simples  
✅ **Prédiction en temps réel** - Résultats instantanés avec probabilités  
✅ **Prédiction en batch** - Upload et traitement de fichiers CSV  
✅ **Visualisation** - Graphiques et statistiques des résultats  
✅ **Pipeline ML robuste** - Preprocessing automatique et scaling des données  
✅ **API REST** - Endpoints FastAPI pour intégration facile

## 📋 Structure du Projet

```
.
├── main.py                 # Application FastAPI principale
├── train_model.py          # Script d'entraînement du modèle
├── churn_pipeline.pkl      # Modèle sauvegardé (KNN) 
├── requirements.txt        # Dépendances Python
│
└── templates/              # Templates HTML
    ├── index.html          # Page d'accueil
    ├── about.html          # Page À propos
    ├── form.html           # Formulaire de prédiction individuelle
    ├── batch.html          # Formulaire d'upload CSV
    └── upload_csv.html     # Résultats de prédiction batch
```

## 🛠️ Installation

### Prérequis
- Python 3.13+
- pip ou conda

### Étapes

1. **Cloner le répertoire**
```bash
git clone https://github.com/hdmanoach/Churn-Prediction-AI.git
cd "Churn-Prediction-AI"
```

2. **Créer un environnement virtuel**
```bash
python3 -m venv env
source env/bin/activate  # Linux/Mac
# ou
env\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### Lancer l'application
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

L'application sera accessible à : `http://localhost:8000`

### Routes disponibles

| Route | Méthode | Description |
|-------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/about` | GET | Page d'informations |
| `/single` | GET | Formulaire de prédiction simple |
| `/batch` | GET | Page d'upload CSV |
| `/predict` | POST | Prédiction d'un client unique |
| `/predict_csv` | POST | Prédiction batch depuis CSV |

## 🤖 Modèle Machine Learning

### Type : KNN (K-Nearest Neighbors)
- **k = 10** voisins
- **Algorithme** : SupervisedClassification
- **Données** : Telco Customer Churn Dataset

### Pipeline
```
1. Données brutes
   ↓
2. Nettoyage (suppression NaN)
   ↓
3. Préprocessing :
   - Scaling numérique (StandardScaler)
   - Encodage catégorique (OneHotEncoder)
   ↓
4. Classification KNN
   ↓
5. Prédiction : Churn (Oui/Non)
```

### Colonnes d'entrée
- `gender` - Genre du client
- `SeniorCitizen` - Statut senior (0/1)
- `tenure` - Durée d'abonnement (mois)
- `MonthlyCharges` - Frais mensuels ($)
- `TotalCharges` - Frais totaux ($)
- Et 15+ autres caractéristiques démographiques et de service

## 📊 Exemple d'utilisation

### Prédiction simple
```
Input:
  - Genre : Male
  - Senior Citizen : 0
  - Tenure : 12 mois
  - Charges mensuelles : $65.50

Output:
  ✅ Le client reste
  Probabilité de churn : 15.43%
```

### Prédiction batch
1. Préparer un CSV avec les colonnes requises
2. Uploader via `/batch`
3. Obtenir un rapport avec :
   - Nombre de clients qui partent/restent
   - Pourcentages
   - Tableau complet des résultats

## 📦 Dépendances principales

- **FastAPI** - Framework web moderne
- **Uvicorn** - Serveur ASGI
- **scikit-learn** - Machine Learning
- **Pandas** - Manipulation de données
- **Jinja2** - Templates HTML
- **Joblib** - Sérialisation de modèles

## 📈 Entraînement du modèle

Réentraîner le modèle avec :
```bash
python train_model.py
```

Cela téléchargera le dataset Telco, entraînera le pipeline KNN et sauvegardera `churn_pipeline.pkl`.

## 🔍 Fichiers clés

### [main.py](main.py)
Application FastAPI avec tous les endpoints et logique de prédiction.

### [train_model.py](train_model.py)
Script d'entraînement du modèle KNN avec préprocessing complet.

## 📝 Format du CSV pour prediction batch

```csv
gender,SeniorCitizen,tenure,MonthlyCharges,TotalCharges,Partner,Dependents,PhoneService,...
Male,0,12,65.50,786.00,No,No,Yes,...
Female,1,24,99.99,2399.76,Yes,Yes,No,...
```

## 🎓 Cas d'usage

- **Rétention client** : Identifier les clients à risque
- **Stratégie marketing** : Cibler les interventions
- **Analyse business** : Comprendre les patterns de churn
- **Rapport managerial** : Export batch des prédictions

## 🔐 Notes de sécurité

- Valider toujours les entrées utilisateur
- Limiter la taille des fichiers uploadés
- Implémenter l'authentification pour la production
- Utiliser HTTPS en production

## 📞 Support

Pour les bugs ou questions, ouvrez une issue sur GitHub.

## 📄 Licence

MIT License - Libre d'utilisation

## 👨‍💻 Auteur

HOSSOU DODO Manoach 

---

⭐ **N'oublie pas de star si c'est utile !**
