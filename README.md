MLOps - TP1 : Prédiction du churn client
Description
Ce projet constitue une première expérience de Machine Learning avant l'introduction des pratiques MLOps. Il consiste à générer un jeu de données de 500 clients fictifs, à explorer et préparer ces données, puis à entraîner et évaluer un modèle de classification pour prédire le churn (départ d'un client).
Le TP met également en évidence les limites d'un projet ML sans suivi des versions des données et des modèles.
Objectifs
- Générer et enregistrer un dataset de clients fictifs.
- Explorer les données avec pandas.
- Préparer les variables explicatives (features) et la variable cible (target).
- Séparer les données en ensembles d'entraînement (80 %) et de test (20 %).
- Entraîner un modèle RandomForestClassifier.
- Évaluer le modèle avec l'accuracy et le rapport de classification.
- Sauvegarder le modèle avec joblib.
- Comprendre les problèmes de reproductibilité et de traçabilité lorsque les versions ne sont pas suivies.
Technologies utilisées
- Python
- pandas
- NumPy
- scikit-learn
- Jupyter
- joblib
- Visual Studio Code (VS Code)
Installation
1. Créer le dossier du projet
Exemple sous Windows :
mkdir D:\MLOps\TP1
cd D:\MLOps\TP1
Ouvrez ensuite ce dossier dans VS Code.
2. Créer et activer l'environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1
Si PowerShell bloque l'activation du script, exécutez cette commande dans le terminal courant, puis réessayez l'activation :
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
3. Installer les dépendances
python -m pip install pandas numpy scikit-learn jupyter joblib
Pour vérifier l'installation :
pip list
Déroulement du TP
Étape 1 — Génération des données
Le script generate_data.py crée 500 clients fictifs avec les colonnes suivantes :
- age : âge du client ;
- salaire : salaire mensuel simulé ;
- abonnement : type d'abonnement (Basique, Standard ou Premium) ;
- anciennete : ancienneté en années ;
- consommation : score d'utilisation du service ;
- churn : variable cible (0 = le client reste, 1 = le client part).
La cible est générée avec une règle simplifiée : un client dont la consommation est inférieure à 30 et l'ancienneté inférieure à 2 ans est marqué comme ayant quitté le service.
Exécutez le script :
python generate_data.py
Le fichier churn_data.csv est créé dans le dossier courant.
Étape 2 — Exploration des données
Dans le notebook churn_model.ipynb, chargez le fichier CSV et vérifiez notamment :
import pandas as pd

df = pd.read_csv("churn_data.csv")
df.head()
df.shape
df.describe()
df.isnull().sum()
df["churn"].value_counts()
Le dataset attendu contient 500 lignes et 6 colonnes.
Étape 3 — Préparation des données
La colonne abonnement est convertie en valeurs numériques :
- Basique → 0
- Standard → 1
- Premium → 2
Les cinq features sont age, salaire, abonnement, anciennete et consommation. La target est churn.
Les données sont séparées avec train_test_split : 80 % pour l'entraînement (400 clients) et 20 % pour le test (100 clients), avec random_state=42.
Étape 4 — Entraînement et évaluation
Le modèle utilisé est une forêt aléatoire (RandomForestClassifier) configurée initialement avec :
- n_estimators=100
- max_depth=5
- random_state=42
Le modèle est entraîné sur les données d'entraînement puis évalué sur les données de test à l'aide de :
- Accuracy : proportion de prédictions correctes ;
- Classification report : précision (precision), rappel (recall), F1-score et support pour chaque classe.
Remarque pédagogique : une accuracy de 1.0 peut être obtenue parce que la cible churn est construite directement à partir de consommation et anciennete, qui font partie des features. Cela illustre une forme de fuite de données (data leakage) et ne signifie pas nécessairement que le modèle serait aussi performant sur de vraies données.
Étape 5 — Sauvegarde du modèle
Le modèle entraîné est sauvegardé avec joblib :
import joblib

joblib.dump(model, "model.pkl")
Le fichier model.pkl est écrasé à chaque nouvelle sauvegarde : les versions précédentes ne sont pas conservées.
Étape 6 — Trois essais
Le TP demande de comparer trois essais en modifiant les hyperparamètres ou les features. Notez les accuracies obtenues et répondez aux questions de réflexion du sujet.
À la fin de l'exercice, model.pkl doit contenir le modèle du troisième essai. Sans suivi des versions, il devient difficile de retrouver ou de comparer précisément les modèles précédents.
Structure indicative du projet
TP1/
├── venv/                 # Environnement virtuel (non nécessaire à partager)
├── generate_data.py      # Génération du dataset
├── churn_data.csv        # Données générées
├── churn_model.ipynb     # Exploration, préparation et entraînement
├── model.pkl             # Modèle entraîné sauvegardé
└── README.md             # Documentation du projet
Les fichiers listés ci-dessus sont ceux attendus dans le déroulement du TP ; adaptez cette structure aux fichiers réellement présents dans votre dossier.
Validation finale
- [ ] L'environnement virtuel a été créé et activé.
- [ ] Les dépendances sont installées.
- [ ] churn_data.csv contient 500 clients fictifs.
- [ ] Le notebook s'exécute sans erreur.
- [ ] Les trois essais ont été réalisés et leurs accuracies notées.
- [ ] model.pkl contient le modèle du troisième essai.
- [ ] Les trois questions de réflexion ont été traitées.
Limites observées et suite du module
Ce premier TP met en évidence plusieurs limites :
- les données ne sont pas versionnées ;
- une sauvegarde peut écraser un modèle précédent ;
- les performances ne sont pas suivies de manière organisée entre les essais ;
- les différences d'environnement peuvent compliquer la reproduction des résultats.
Les TP suivants introduiront progressivement des outils comme Git pour le suivi des versions et MLflow pour le suivi des expériences.
