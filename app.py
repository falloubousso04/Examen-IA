"""
API d'évaluation du risque de crédit.

Endpoints :
    GET  /                  plateforme web (index.html)
    GET  /health            état de l'API
    GET  /meta              listes pour les formulaires (agences, secteurs, types de crédit) + repères
    GET  /rapport           statistiques des données et performances du modèle
    POST /predict           évalue une demande de crédit
    POST /predict/batch     évalue un fichier Excel / CSV de crédits

Lancement :
    uvicorn app:app --port 8000
Documentation interactive : http://localhost:8000/docs
"""
import io
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from credit_lib import RETARDS_SAINS, famille, niveau_risque, normaliser, preparer, renommer_colonnes

DOSSIER = Path(__file__).parent
ARTEFACTS = DOSSIER / 'model_artifacts'

app = FastAPI(
    title="API d'évaluation du risque de crédit",
    description="Prédit la probabilité qu'un crédit tombe en retard de plus de 30 jours (PAR 30).",
    version='1.0.0',
)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# Chargement du modèle et des métadonnées au démarrage
_systeme = joblib.load(ARTEFACTS / 'modele.joblib')
MODELE, SEUIL = _systeme['pipeline'], _systeme['seuil']
META = json.loads((ARTEFACTS / 'meta.json').read_text(encoding='utf-8'))
RAPPORT = json.loads((ARTEFACTS / 'rapport.json').read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Évaluation d'une demande
# ---------------------------------------------------------------------------
class DemandeCredit(BaseModel):
    agence: str = Field(..., examples=['CEC_B'])
    secteur: str = Field(..., examples=['COMMERCE'])
    type_credit: str = Field(..., examples=['CRÉDITS TRESORERIE FEMME AFSSEF  CT DE + DE 6 À 12 MOIS AU +'])
    age: int = Field(..., ge=18, le=100, description="Âge de l'emprunteur")
    nb_credits_anterieurs: Optional[int] = Field(None, ge=0, le=50)
    montant: float = Field(..., gt=0, description='Montant demandé en FCFA')
    mois_ouverture: int = Field(..., ge=1, le=12)


def json_safe(v):
    if pd.isna(v):
        return None
    if isinstance(v, np.generic):
        v = v.item()
    return round(v, 4) if isinstance(v, float) else v


def repere(categorie, cle):
    r = META['reperes'][categorie].get(cle)
    return {'nom': cle, **r} if r else {'nom': cle, 'taux': None, 'n': 0}


@app.post('/predict')
def predict(d: DemandeCredit):
    ligne = pd.DataFrame([{
        'AGENCE': d.agence, 'SECTEUR': d.secteur, 'TYPE_CREDIT': d.type_credit, 'AGE': d.age,
        'NB_CREDITS_ANT': d.nb_credits_anterieurs, 'MONTANT_DEBOURSE': d.montant,
        'MOIS_OUVERTURE': d.mois_ouverture}])
    X = preparer(ligne)
    proba = float(MODELE.predict_proba(X)[0, 1])
    niveau = str(niveau_risque([proba], SEUIL)[0])
    type_norm = normaliser(d.type_credit)
    return {
        'probabilite_defaut': round(proba, 4),
        'niveau_risque': niveau,
        'decision': {'ÉLEVÉ': 'Étude approfondie : garanties supplémentaires ou refus',
                     'MODÉRÉ': 'Avis favorable sous conditions : suivi rapproché des remboursements',
                     'FAIBLE': 'Avis favorable'}[niveau],
        'seuil': SEUIL,
        'variables': {k: json_safe(v) for k, v in X.iloc[0].items()},
        'reperes': {
            'type_credit': repere('type_credit', type_norm),
            'famille': repere('famille', famille(type_norm)),
            'agence': repere('agence', d.agence),
            'secteur': repere('secteur', d.secteur),
            'global': META['taux_global'],
        },
    }


# ---------------------------------------------------------------------------
# Analyse d'un fichier entier
# ---------------------------------------------------------------------------
COLONNES_REQUISES = ['AGENCE', 'SECTEUR', 'TYPE_CREDIT', 'AGE', 'MONTANT_DEBOURSE']
MAX_LIGNES = 50000


def lire_fichier(nom, contenu):
    if nom.lower().endswith(('.xlsx', '.xls')):
        try:
            return pd.read_excel(io.BytesIO(contenu))
        except Exception:
            raise HTTPException(400, "Fichier Excel illisible.")
    for encodage in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            texte = contenu.decode(encodage)
        except UnicodeDecodeError:
            continue
        for sep in (';', ',', '\t'):
            try:
                df = pd.read_csv(io.StringIO(texte), sep=sep)
            except Exception:
                continue
            if len(df.columns) >= len(COLONNES_REQUISES):
                return df
    raise HTTPException(400, 'Fichier illisible : envoyez un fichier Excel (.xlsx) ou CSV.')


@app.post('/predict/batch')
async def predict_batch(file: UploadFile = File(...)):
    df = renommer_colonnes(lire_fichier(file.filename or '', await file.read()))
    manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if manquantes:
        raise HTTPException(400, f"Colonnes manquantes : {', '.join(manquantes)}. "
                                 'Le fichier doit avoir le même format que DataSet_Credit.xlsx.')
    if len(df) > MAX_LIGNES:
        raise HTTPException(413, f'Fichier trop volumineux ({len(df)} lignes, maximum {MAX_LIGNES}).')

    n_brut = len(df)
    df = df.dropna(subset=COLONNES_REQUISES)
    X = preparer(df)
    df, X = df[X['LOG_MONTANT'].notna()], X[X['LOG_MONTANT'].notna()]
    if df.empty:
        raise HTTPException(400, 'Aucune ligne exploitable après nettoyage.')

    proba = MODELE.predict_proba(X)[:, 1]
    niveaux = niveau_risque(proba, SEUIL)
    montants = pd.to_numeric(df['MONTANT_DEBOURSE'])
    numeros = df['NUMERO_CREDIT'].astype(str) if 'NUMERO_CREDIT' in df else pd.Series(range(1, len(df) + 1), index=df.index).astype(str)

    lignes = [[n, a, s, t, float(m), round(float(p), 4), str(nv)]
              for n, a, s, t, m, p, nv in zip(numeros, df['AGENCE'].astype(str), df['SECTEUR'].astype(str),
                                              df['TYPE_CREDIT'].astype(str), montants, proba, niveaux)]
    lignes.sort(key=lambda l: -l[5])

    resultat = {
        'fichier': file.filename,
        'n_lignes_fichier': int(n_brut),
        'n_analyses': int(len(df)),
        'montant_total': float(montants.sum()),
        'montant_risque_eleve': float(montants[niveaux == 'ÉLEVÉ'].sum()),
        'repartition': {nv: int((niveaux == nv).sum()) for nv in ['ÉLEVÉ', 'MODÉRÉ', 'FAIBLE']},
        'proba_moyenne': float(proba.mean()),
        'seuil': SEUIL,
        'colonnes': ['numero', 'agence', 'secteur', 'type_credit', 'montant', 'probabilite', 'niveau'],
        'lignes': lignes,
    }

    # Si la colonne RETARD est présente, on compare les prédictions à la réalité
    if 'RETARD' in df.columns:
        reel_brut = df['RETARD']
        valides = reel_brut.notna().to_numpy()
        if valides.any():
            reel = (~reel_brut[valides].isin(RETARDS_SAINS)).astype(int).to_numpy()
            pred = (proba[valides] >= SEUIL).astype(int)
            vp, fp = int(((reel == 1) & (pred == 1)).sum()), int(((reel == 0) & (pred == 1)).sum())
            fn, vn = int(((reel == 1) & (pred == 0)).sum()), int(((reel == 0) & (pred == 0)).sum())
            resultat['evaluation'] = {
                'n_evalues': int(valides.sum()),
                'accuracy': (vp + vn) / int(valides.sum()),
                'precision': vp / (vp + fp) if vp + fp else 0.0,
                'recall': vp / (vp + fn) if vp + fn else 0.0,
                'confusion': {'vrais_negatifs': vn, 'faux_positifs': fp, 'faux_negatifs': fn, 'vrais_positifs': vp},
                'taux_reel': float(reel.mean()),
            }
    return resultat


# ---------------------------------------------------------------------------
# Informations
# ---------------------------------------------------------------------------
@app.get('/health')
def health():
    return {'status': 'ok', 'modele': RAPPORT['final']['nom'], 'entraine_le': META['entraine_le']}


@app.get('/meta')
def meta():
    return META


@app.get('/rapport')
def rapport():
    return RAPPORT


@app.get('/', include_in_schema=False)
def accueil():
    return FileResponse(DOSSIER / 'index.html')
