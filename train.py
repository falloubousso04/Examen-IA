"""
Pipeline complet : chargement -> nettoyage -> comparaison de 5 modèles ->
optimisation -> choix du seuil -> évaluation -> sauvegarde dans model_artifacts/.

    python train.py                      # utilise DataSet_Credit.xlsx
    python train.py autre_fichier.xlsx

Produit :
    model_artifacts/modele.joblib   pipeline scikit-learn (prétraitement + modèle) et seuil
    model_artifacts/meta.json       listes des agences, secteurs, types, repères historiques
    model_artifacts/rapport.json    statistiques des données et performances mesurées
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from credit_lib import ORDRE_RETARDS, VARS_CAT, VARS_NUM, charger_dataset, famille, normaliser, preparer

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
DOSSIER = Path(__file__).parent
ARTEFACTS = DOSSIER / 'model_artifacts'


def preprocesseur(vars_num=VARS_NUM, vars_cat=VARS_CAT):
    return ColumnTransformer([
        ('num', Pipeline([('imputation', SimpleImputer(strategy='median')),
                          ('standardisation', StandardScaler())]), vars_num),
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20), vars_cat),
    ])


def modeles_candidats(ratio):
    return {
        'Régression logistique': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'Arbre de décision': DecisionTreeClassifier(max_depth=8, min_samples_leaf=10,
                                                    class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight='balanced',
                                                n_jobs=-1, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=ratio,
                                 eval_metric='logloss', random_state=RANDOM_STATE),
    }


def taux_par(df, colonne, min_effectif=1):
    g = df.groupby(colonne)['RISQUE'].agg(['mean', 'size'])
    g = g[g['size'] >= min_effectif].sort_values('mean', ascending=False)
    return [{'nom': str(k), 'taux': round(float(r['mean']), 4), 'n': int(r['size'])} for k, r in g.iterrows()]


def points_roc(y, p, n=80):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]


def metriques(y, pred):
    return {'accuracy': accuracy_score(y, pred), 'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred), 'f1': f1_score(y, pred)}


def main(fichier):
    print(f'Chargement de {fichier}')
    df = charger_dataset(fichier)
    n_sans_cible = df.attrs['lignes_sans_cible']
    X, y = preparer(df), df['RISQUE']
    df['FAMILLE'], df['DUREE'] = X['FAMILLE'], X['DUREE']
    print(f'{len(df)} crédits, {y.mean():.1%} à risque (retard > 30 jours)')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # 1. Comparaison des modèles (validation croisée + jeu de test)
    print('\nComparaison des modèles')
    comparaison, roc = [], {}
    for nom, modele in modeles_candidats(ratio).items():
        pipe = Pipeline([('prep', preprocesseur()), ('modele', modele)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        p = pipe.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        ligne = {'nom': nom, 'auc_cv': cv_auc.mean(), 'auc_cv_std': cv_auc.std(),
                 'auc': roc_auc_score(y_test, p), **metriques(y_test, (p >= 0.5).astype(int))}
        comparaison.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ligne.items()})
        roc[nom] = points_roc(y_test, p)
        print(f'  {nom:22s} AUC CV = {cv_auc.mean():.4f}   AUC test = {ligne["auc"]:.4f}')

    # 2. Optimisation du Random Forest
    print('\nOptimisation du Random Forest (GridSearch)')
    grille = {'modele__n_estimators': [200], 'modele__max_depth': [None, 20],
              'modele__min_samples_leaf': [1, 3], 'modele__max_features': ['sqrt', 0.3]}
    recherche = GridSearchCV(
        Pipeline([('prep', preprocesseur()),
                  ('modele', RandomForestClassifier(class_weight='balanced', n_jobs=-1,
                                                    random_state=RANDOM_STATE))]),
        grille, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE), scoring='roc_auc', n_jobs=-1)
    recherche.fit(X_train, y_train)
    print(f'  {recherche.best_params_}')

    # 3. Calibrage des probabilités : la pondération des classes gonfle les probabilités
    #    du Random Forest ; le calibrage isotonique les ramène à des fréquences réelles
    #    (une probabilité affichée de 20 % correspond à environ 20 % de crédits en retard)
    params_rf = {k.replace('modele__', ''): v for k, v in recherche.best_params_.items()}
    meilleur = Pipeline([('prep', preprocesseur()), ('modele', CalibratedClassifierCV(
        RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE, **params_rf),
        cv=5, method='isotonic'))])
    meilleur.fit(X_train, y_train)

    # 4. Seuil de décision : maximise le F1, choisi sur les prédictions hors-pli
    #    du jeu d'apprentissage (le jeu de test reste intact pour l'évaluation)
    p_oof = cross_val_predict(meilleur, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    seuils = np.round(np.arange(0.10, 0.90, 0.01), 2)
    seuil = float(seuils[int(np.argmax([f1_score(y_train, (p_oof >= s).astype(int)) for s in seuils]))])

    # 5. Évaluation finale sur le jeu de test
    p_test = meilleur.predict_proba(X_test)[:, 1]
    pred = (p_test >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    final = {'nom': 'Random Forest optimisé et calibré', 'parametres': params_rf,
             'proba_moyenne': round(float(p_test.mean()), 4), 'taux_reel_test': round(float(y_test.mean()), 4),
             'brier': round(brier_score_loss(y_test, p_test), 4),
             'seuil': seuil, 'auc': round(roc_auc_score(y_test, p_test), 4),
             **{k: round(v, 4) for k, v in metriques(y_test, pred).items()},
             'confusion': {'vrais_negatifs': int(tn), 'faux_positifs': int(fp),
                           'faux_negatifs': int(fn), 'vrais_positifs': int(tp)},
             'n_test': int(len(y_test)), 'roc': points_roc(y_test, p_test),
             'courbe_seuil': [{'seuil': float(s), **{k: round(v, 4) for k, v in
                                                     metriques(y_test, (p_test >= s).astype(int)).items()}}
                              for s in seuils[::2]]}
    print(f'\nModèle final : AUC test = {final["auc"]:.4f}, seuil = {seuil:.2f}, '
          f'précision = {final["precision"]:.3f}, rappel = {final["recall"]:.3f}')

    # Importance des variables (regroupée par variable d'origine)
    noms = meilleur.named_steps['prep'].get_feature_names_out()
    calibres = meilleur.named_steps['modele'].calibrated_classifiers_
    imp = pd.Series(np.mean([c.estimator.feature_importances_ for c in calibres], axis=0), index=noms)
    origine = {n: next(v for v in VARS_NUM + VARS_CAT
                       if n.split('__', 1)[1] == v or n.split('__', 1)[1].startswith(v + '_')) for n in noms}
    imp_var = imp.groupby(origine).sum().sort_values(ascending=False)
    imp_detail = imp.sort_values(ascending=False).head(12)
    final['importances'] = [{'nom': k, 'valeur': round(float(v), 4)} for k, v in imp_var.items()]
    def libelle(n):
        v, nom = origine[n], n.split('__', 1)[1]
        return v if nom == v else f'{v} = {nom[len(v) + 1:]}'
    final['importances_detail'] = [{'nom': libelle(k), 'valeur': round(float(v), 4)} for k, v in imp_detail.items()]

    # 6. Démonstration de la fuite de données : même modèle avec SOLDE et ANNEE
    fuite = X.assign(ANNEE_OUVERTURE=df['ANNEE_OUVERTURE'],
                     TAUX_RESTANT=df['SOLDE_CREDIT'] / df['MONTANT_DEBOURSE'])
    Xf_tr, Xf_te = fuite.loc[X_train.index], fuite.loc[X_test.index]
    pipe_fuite = Pipeline([('prep', preprocesseur(VARS_NUM + ['ANNEE_OUVERTURE', 'TAUX_RESTANT'])),
                           ('modele', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                             n_jobs=-1, random_state=RANDOM_STATE))])
    auc_fuite = roc_auc_score(y_test, pipe_fuite.fit(Xf_tr, y_train).predict_proba(Xf_te)[:, 1])
    print(f'AUC avec variables de fuite (non utilisées) : {auc_fuite:.4f}')

    # 7. Réentraînement sur toutes les données avant déploiement
    meilleur.fit(X, y)

    # Sauvegarde
    ARTEFACTS.mkdir(exist_ok=True)
    joblib.dump({'pipeline': meilleur, 'seuil': seuil}, ARTEFACTS / 'modele.joblib', compress=3)

    df['TYPE_NORM'] = X['TYPE_CREDIT']
    libelles = df.groupby('TYPE_NORM')['TYPE_CREDIT'].agg(lambda s: s.value_counts().index[0])
    types = [{'libelle': libelles[t], 'famille': famille(t), 'n': int(n)}
             for t, n in df['TYPE_NORM'].value_counts().items()]
    reference = lambda col: {r['nom']: {'taux': r['taux'], 'n': r['n']} for r in taux_par(df, col)}
    meta = {
        'agences': sorted(df['AGENCE'].unique().tolist()),
        'secteurs': df['SECTEUR'].value_counts().index.tolist(),
        'types_credit': types,
        'seuil': seuil,
        'taux_global': round(float(y.mean()), 4),
        'reperes': {'agence': reference('AGENCE'), 'secteur': reference('SECTEUR'),
                    'type_credit': reference('TYPE_NORM'), 'famille': reference('FAMILLE')},
        'age_median': float(X['AGE'].median()),
        'entraine_le': datetime.now().strftime('%d/%m/%Y %H:%M'),
    }

    retards = df['RETARD'].value_counts()
    tranches = pd.cut(df['MONTANT_DEBOURSE'], [0, 200_000, 500_000, 1_000_000, 5_000_000, np.inf],
                      labels=['< 200 k', '200 k - 500 k', '500 k - 1 M', '1 M - 5 M', '> 5 M'])
    rapport = {
        'donnees': {
            'fichier': Path(fichier).name, 'n': int(len(df)), 'n_risque': int(y.sum()),
            'taux_risque': round(float(y.mean()), 4), 'n_agences': int(df['AGENCE'].nunique()),
            'montant_total': float(df['MONTANT_DEBOURSE'].sum()),
            'encours_total': float(df['SOLDE_CREDIT'].sum()),
            'encours_risque': float(df.loc[df['RISQUE'] == 1, 'SOLDE_CREDIT'].sum()),
            'retards': [{'nom': r, 'n': int(retards.get(r, 0))} for r in ORDRE_RETARDS],
            'par_agence': taux_par(df, 'AGENCE'),
            'par_famille': taux_par(df, 'FAMILLE'),
            'par_secteur': taux_par(df, 'SECTEUR', min_effectif=30),
            'par_montant': [{'nom': str(k), 'taux': round(float(v['mean']), 4), 'n': int(v['size'])}
                            for k, v in df.groupby(tranches, observed=True)['RISQUE'].agg(['mean', 'size']).iterrows()],
            'nettoyage': {'cible_manquante': int(n_sans_cible), 'ages_aberrants': int((pd.to_numeric(df['AGE']) < 18).sum()),
                          'nb_credits_ant_manquants': int(df['NB_CREDITS_ANT'].isna().sum()),
                          'types_avant': int(df['TYPE_CREDIT'].nunique()),
                          'types_apres': int(X['TYPE_CREDIT'].nunique())},
        },
        'comparaison': comparaison,
        'roc': roc,
        'final': final,
        'fuite': {'auc_avec_fuite': round(auc_fuite, 4), 'auc_sans_fuite': final['auc']},
        'variables': VARS_NUM + VARS_CAT,
        'entraine_le': meta['entraine_le'],
    }

    (ARTEFACTS / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    (ARTEFACTS / 'rapport.json').write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nArtefacts enregistrés dans {ARTEFACTS}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DOSSIER / 'DataSet_Credit.xlsx')
