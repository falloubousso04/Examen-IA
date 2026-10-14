# Évaluation du risque de crédit — API & plateforme web

Système d'intelligence artificielle qui estime, au moment de la demande, la
probabilité qu'un crédit tombe en **retard de remboursement de plus de 30 jours**
(indicateur PAR 30, référence en microfinance). Une API REST expose le modèle,
et une plateforme web permet d'évaluer une demande, d'analyser un portefeuille
complet et de consulter les performances du modèle.

## Fonctionnalités de la plateforme

| Onglet | Rôle |
|---|---|
| **Évaluer une demande** | Formulaire (agence, secteur, type de crédit, montant, âge…) → probabilité de défaut, niveau de risque (faible / modéré / élevé), recommandation et repères historiques |
| **Analyser un fichier** | Import d'un fichier Excel/CSV : chaque crédit est scoré, classé du plus au moins risqué, filtrable et exportable en CSV. Si la colonne `RETARD` est présente, les prédictions sont comparées à la réalité |
| **Tableau de bord** | Statistiques du portefeuille : taux de risque par agence, famille de crédit, secteur, tranche de montant |
| **Modèle** | Comparaison des 5 modèles (courbes ROC, AUC), matrice de confusion, choix du seuil, importance des variables, fuite de données évitée |

## Données et préparation

Fichier `DataSet_Credit.xlsx` : 15 189 crédits, 5 agences, 12 colonnes.

- **Cible :** `RISQUE = 1` si `RETARD` > 30 jours (13,5 % des crédits, classes déséquilibrées).
- **Nettoyage :** suppression de la ligne sans cible ; âge aberrant (< 18 ans) mis
  en valeur manquante ; 2 010 valeurs manquantes de « Nombre de crédit antérieur »
  imputées par la médiane ; harmonisation des libellés de type de crédit (accents,
  majuscules, espaces).
- **Nouvelles variables :** `DUREE` (court / moyen / long terme) et `FAMILLE`
  (trésorerie, personnel, entente PAM/DER…), déduites du libellé ; `LOG_MONTANT`.
- **Fuite de données évitée :** `SOLDE_CREDIT` (connu seulement après l'octroi) et
  `ANNEE_OUVERTURE` (un crédit ancien encore ouvert est presque toujours en défaut)
  sont exclus. Avec ces variables, l'AUC monterait à 0,99, mais le modèle serait
  inutilisable pour un nouveau client.

L'exploration et le nettoyage sont détaillés pas à pas dans le notebook
`Projet_Risque_Credit.ipynb` (exécutable sur Google Colab).

## Résultats

5 modèles comparés en validation croisée (5 plis), puis sur un jeu de test de 20 %
jamais vu à l'entraînement :

| Modèle | AUC CV | AUC test |
|---|---|---|
| **Random Forest** (retenu) | 0,894 | **0,924** |
| XGBoost | 0,891 | 0,923 |
| Gradient Boosting | 0,896 | 0,923 |
| Régression logistique | 0,876 | 0,912 |
| Arbre de décision | 0,870 | 0,909 |

Random Forest optimisé par GridSearch, puis **calibré** (méthode isotonique) pour que
les probabilités affichées correspondent à des fréquences réelles : probabilité moyenne
prédite 14,2 % pour un taux réel de 13,5 % sur le test. Seuil de décision 0,36
(maximise le F1, choisi sur les données d'apprentissage) :
**exactitude 94,7 %**, précision 95,3 %, rappel 63,7 % sur le jeu de test.

Facteur le plus déterminant : le **type de crédit**. Les crédits « entente PAM/DER »
sont en retard dans près de 88 % des cas.

## Contenu

```
├── app.py                  # API FastAPI (évaluation, analyse de fichier, rapport)
├── credit_lib.py           # nettoyage et préparation des variables, partagés entraînement / API
├── train.py                # pipeline complet : nettoyage → comparaison → optimisation → évaluation
├── index.html              # plateforme web, servie par l'API à la racine
├── model_artifacts/
│   ├── modele.joblib       # pipeline scikit-learn (prétraitement + Random Forest calibré) et seuil
│   ├── meta.json           # listes pour les formulaires et repères historiques
│   └── rapport.json        # statistiques des données et performances mesurées
├── Projet_Risque_Credit.ipynb   # exploration et nettoyage (Google Colab)
├── requirements.txt
└── Dockerfile
```

Le fichier de données n'est **pas** versionné (numéros de crédit et de membre).
Le modèle déjà entraîné est fourni : la plateforme fonctionne sans lui.

## Lancer la plateforme

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

- Plateforme : http://localhost:8000
- Documentation interactive de l'API : http://localhost:8000/docs

Exemple d'appel :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"agence": "CEC_B", "secteur": "COMMERCE",
       "type_credit": "CREDITS PERSONNEL A MT DE+1 AN A 2 AN AU +",
       "age": 45, "nb_credits_anterieurs": 3, "montant": 500000, "mois_ouverture": 3}'
```

Réponse (extrait) :

```json
{
  "probabilite_defaut": 0.0759,
  "niveau_risque": "FAIBLE",
  "decision": "Avis favorable",
  "seuil": 0.36,
  "reperes": { "type_credit": {"taux": 0.053, "n": 640}, "agence": {"taux": 0.084, "n": 5388}, "global": 0.135 }
}
```

Niveaux de risque : **élevé** si la probabilité ≥ seuil (0,36), **modéré** si elle
est ≥ la moitié du seuil, **faible** sinon.

## Analyser un fichier entier

```bash
curl -X POST http://localhost:8000/predict/batch -F "file=@DataSet_Credit.xlsx"
```

Format attendu : celui de `DataSet_Credit.xlsx` (Excel ou CSV). Colonnes requises :
`AGENCE`, `SECTEUR ACTIVITE`, `TYPE DE CREDIT`, `Age`, `MONTANT_DEBOURSE`.
Colonnes facultatives : `Nombre de Crédit antérieur`, `MOIS OUVERTURE`,
`NUMERO_CREDIT`, et `RETARD` (déclenche la comparaison avec la réalité).

> Analyser le fichier qui a servi à l'entraînement donne des scores optimistes,
> puisque le modèle a déjà vu ces crédits. Seule l'évaluation sur le jeu de test
> (onglet **Modèle**) mesure la vraie performance.

## Réentraîner le modèle

Déposer `DataSet_Credit.xlsx` à la racine, puis :

```bash
python train.py
```

Le script nettoie les données, compare les 5 modèles, optimise et calibre le
Random Forest, choisit le seuil, évalue sur le jeu de test, réentraîne sur toutes les données et
régénère `model_artifacts/` (environ 2 minutes).

## Déployer avec Docker

```bash
docker build -t risque-credit .
docker run -d -p 8000:8000 risque-credit
```

## Limites

- Les données ne contiennent ni revenus ni garanties de l'emprunteur : le modèle
  est un **outil d'aide à la décision**, pas un remplaçant de l'agent de crédit.
- Une grande partie du pouvoir prédictif vient du type de produit ; le modèle
  distingue moins bien les clients au sein d'un même produit.
- Rappel de 64 % : environ un tiers des crédits à risque ne sont pas signalés
  comme « élevé ». Baisser le seuil détecte plus de risques, au prix de davantage
  de fausses alertes (voir la courbe du choix du seuil dans l'onglet Modèle).
- Découpe aléatoire : une découpe chronologique (entraîner sur les anciens crédits,
  tester sur les récents) mesurerait mieux la stabilité dans le temps.
- API sans authentification : ajouter une clé API avant toute exposition publique.
