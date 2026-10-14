"""
Fonctions de nettoyage et de préparation des données, partagées par
l'entraînement (train.py) et l'API (app.py) : entraînement et prédiction
transforment donc les données exactement de la même façon.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

# Nom court de chaque colonne du fichier source, dans l'ordre de DataSet_Credit.xlsx
COLONNES = ['NUMERO_CREDIT', 'AGENCE', 'SECTEUR', 'TYPE_CREDIT', 'NUMERO_MEMBRE',
            'ANNEE_OUVERTURE', 'MOIS_OUVERTURE', 'AGE', 'NB_CREDITS_ANT',
            'SOLDE_CREDIT', 'MONTANT_DEBOURSE', 'RETARD']

# Variables connues au moment de la demande de crédit (pas de fuite de données)
VARS_NUM = ['AGE', 'NB_CREDITS_ANT', 'LOG_MONTANT', 'MOIS_OUVERTURE']
VARS_CAT = ['AGENCE', 'SECTEUR', 'TYPE_CREDIT', 'DUREE', 'FAMILLE']

# Un crédit est « à risque » au-delà de 30 jours de retard (indicateur PAR 30)
RETARDS_SAINS = ['0 JOUR', '1 à 30 jours']
ORDRE_RETARDS = ['0 JOUR', '1 à 30 jours', '31 à 60 jours', '61 à 90 jours',
                 '91 à 180 jours', '181 à 3600 jours']


def normaliser(texte):
    """Majuscules, sans accents ni espaces multiples."""
    texte = unicodedata.normalize('NFKD', str(texte)).encode('ascii', 'ignore').decode().upper()
    return re.sub(r'\s+', ' ', texte).strip()


def duree(type_norm):
    if re.search(r'\bLT\b|3 ANS A 10', type_norm): return 'LT 3-10 ans'
    if '2 ANS A 3' in type_norm:                   return 'MT 2-3 ans'
    if '1 AN' in type_norm:                        return 'MT 1-2 ans'
    return 'CT 6-12 mois'


def famille(type_norm):
    for mot, nom in [('ENTENTE', 'ENTENTE PAM/DER'), ('TRESO', 'TRESORERIE'),
                     ('PERSONNEL', 'PERSONNEL'), ('CONSOMMATION', 'CONSOMMATION'),
                     ('COMMERCIAL', 'COMMERCIAL'), ('HABITAT', 'HABITAT'),
                     ('EQUIPEMENT', 'EQUIPEMENT')]:
        if mot in type_norm:
            return nom
    return 'AUTRE'


def cle_colonne(nom):
    """Clé de comparaison d'un nom de colonne : 'Nombre de Crédit antérieur' -> 'NOMBREDECREDITANTERIEUR'."""
    return re.sub(r'[^A-Z0-9]', '', normaliser(nom))


# Noms de colonnes acceptés dans un fichier importé (clé normalisée -> nom court)
ALIAS_COLONNES = {cle_colonne(a): c for c, alias in {
    'NUMERO_CREDIT':    ['NUMERO_CREDIT', 'NUMERO CREDIT'],
    'AGENCE':           ['AGENCE'],
    'SECTEUR':          ['SECTEUR ACTIVITE', 'SECTEUR'],
    'TYPE_CREDIT':      ['TYPE DE CREDIT', 'TYPE_CREDIT'],
    'NUMERO_MEMBRE':    ['NUMERO_MEMBRE'],
    'ANNEE_OUVERTURE':  ['ANNEE OUVERTURE', 'ANNEE_OUVERTURE'],
    'MOIS_OUVERTURE':   ['MOIS OUVERTURE', 'MOIS_OUVERTURE'],
    'AGE':              ['Age'],
    'NB_CREDITS_ANT':   ['Nombre de Crédit antérieur', 'NB_CREDITS_ANT'],
    'SOLDE_CREDIT':     ['SOLDE_CREDIT'],
    'MONTANT_DEBOURSE': ['MONTANT_DEBOURSE', 'MONTANT'],
    'RETARD':           ['RETARD'],
}.items() for a in alias}


def renommer_colonnes(df):
    return df.rename(columns={c: ALIAS_COLONNES.get(cle_colonne(c), c) for c in df.columns})


def charger_dataset(chemin):
    """Lit DataSet_Credit.xlsx, renomme les colonnes et crée la cible RISQUE."""
    brut = renommer_colonnes(pd.read_excel(chemin))
    df = brut.dropna(subset=['RETARD']).copy()
    df.attrs['lignes_sans_cible'] = len(brut) - len(df)
    df['RISQUE'] = (~df['RETARD'].isin(RETARDS_SAINS)).astype(int)
    return df


def preparer(df):
    """Nettoie les données et construit les variables d'entrée du modèle.

    - âge < 18 ans (impossible) -> valeur manquante (imputée par la médiane dans le pipeline)
    - libellé du type de crédit harmonisé, puis DUREE et FAMILLE déduites du libellé
    - LOG_MONTANT pour réduire l'asymétrie des montants
    """
    age = pd.to_numeric(df['AGE'], errors='coerce')
    montant = pd.to_numeric(df['MONTANT_DEBOURSE'], errors='coerce')
    types = df['TYPE_CREDIT'].map(normaliser)
    nb_ant = df['NB_CREDITS_ANT'] if 'NB_CREDITS_ANT' in df else np.nan
    mois = df['MOIS_OUVERTURE'] if 'MOIS_OUVERTURE' in df else np.nan
    return pd.DataFrame({
        'AGE': age.where(age >= 18),
        'NB_CREDITS_ANT': pd.to_numeric(nb_ant, errors='coerce'),
        'LOG_MONTANT': np.log(montant.where(montant > 0)),
        'MOIS_OUVERTURE': pd.to_numeric(mois, errors='coerce'),
        'AGENCE': df['AGENCE'].astype(str).str.strip(),
        'SECTEUR': df['SECTEUR'].astype(str).str.strip(),
        'TYPE_CREDIT': types,
        'DUREE': types.map(duree),
        'FAMILLE': types.map(famille),
    }, index=df.index)


def niveau_risque(proba, seuil):
    """ÉLEVÉ au-dessus du seuil du modèle, MODÉRÉ au-dessus de la moitié du seuil."""
    proba = np.asarray(proba)
    return np.where(proba >= seuil, 'ÉLEVÉ', np.where(proba >= seuil / 2, 'MODÉRÉ', 'FAIBLE'))
