"""Dados, divisão treino/teste e os dois modelos finais."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.preprocessamento import LimpezaCampanhas, codificador_pos_veiculacao, codificador_pre_veiculacao

CAMINHO_DADOS = Path(__file__).resolve().parents[1] / "data" / "klike_challenge_dataset.csv"
ALVO = "klike_score"
ALPHA = 3.16  # escolhido na validação cruzada, igual nos dois modelos (notebooks/02_modelo.ipynb)


def carregar_treino_teste():
    """Mesma divisão da EDA: 80/20, random_state=42, estratificada por plataforma."""
    df = pd.read_csv(CAMINHO_DADOS)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["platform"])
    return train.drop(columns=ALVO), train[ALVO], test.drop(columns=ALVO), test[ALVO]


def modelo_pre_veiculacao(alpha=ALPHA):
    """Prevê o score só com o que existe antes de a campanha ir ao ar (criativo e contexto)."""
    return make_pipeline(LimpezaCampanhas(), codificador_pre_veiculacao(), Ridge(alpha=alpha))


def modelo_pos_veiculacao(alpha=ALPHA):
    """Acrescenta as métricas de performance observadas depois da veiculação."""
    return make_pipeline(LimpezaCampanhas(), codificador_pos_veiculacao(), Ridge(alpha=alpha))


GRADE_ALPHA = {"ridge__alpha": np.logspace(-2, 3, 21)}
