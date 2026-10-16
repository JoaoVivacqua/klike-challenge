"""Pré-processamento das campanhas: limpeza, novas features e codificação para o modelo."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.limpeza import corrigir_inconsistencias, medianas_por_plataforma

TETO_CTR = 0.15
TETO_ENGAGEMENT = 0.10

# Existem antes de a campanha ir ao ar: decisões de produção do vídeo e de mídia
CRIATIVO_BINARIAS = ["has_hook", "has_face", "has_cta", "has_subtitle"]
CRIATIVO_NUMERICAS = ["video_duration_s", "music_voice_ratio"]
CRIATIVO_CATEGORICAS = ["format", "text_density"]
CONTEXTO_BINARIAS = ["is_retargeting"]
CONTEXTO_CATEGORICAS = ["platform", "category", "objective", "target_audience_age"]

# Só existem depois de a campanha rodar
METRICAS_NUMERICAS = ["log_ctr", "log_cpc", "log_engagement_rate", "watch_ratio", "log_conversion_rate", "log_roas"]
METRICAS_INDICADORAS = ["ctr_no_teto", "engagement_no_teto", "roas_nao_se_aplica"]


class LimpezaCampanhas(BaseEstimator, TransformerMixin):
    """Aplica as correções da EDA (passo 4) e cria as features derivadas.

    No fit, aprende apenas as medianas de ctr e cpc por plataforma, usadas para decidir
    qual coluna está errada quando uma conta não fecha.
    """

    def fit(self, X, y=None):
        self.medianas_ = medianas_por_plataforma(X)
        return self

    def transform(self, X):
        df = corrigir_inconsistencias(X, self.medianas_)
        df["has_subtitle"] = df["has_subtitle"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
        for col in ["has_hook", "has_face", "has_cta", "is_retargeting"]:
            df[col] = df[col].astype(float)

        # vazios recuperáveis pela própria conta (passo 4.3)
        cliques = df["clicks"].replace(0, np.nan)
        df["cpc"] = df["cpc"].fillna(df["spend"] / cliques)
        df["revenue"] = df["revenue"].fillna(df["roas"] * df["spend"])

        df["watch_ratio"] = df["avg_watch_time_s"] / df["video_duration_s"]
        df["conversion_rate"] = df["conversions"] / cliques
        df["ctr_no_teto"] = (df["ctr"] >= TETO_CTR).astype(float)
        df["engagement_no_teto"] = (df["engagement_rate"] >= TETO_ENGAGEMENT).astype(float)

        # receita não medida (passo 4.3): roas zero com conversões não é retorno zero
        nao_medida = (df["roas"] == 0) & (df["conversions"] > 0)
        df["roas_nao_se_aplica"] = nao_medida.astype(float)
        df.loc[nao_medida, "roas"] = np.nan

        # taxas com cauda à direita: log, com as proporções em pontos percentuais
        for col in ["ctr", "engagement_rate", "conversion_rate"]:
            df[f"log_{col}"] = np.log1p(df[col] * 100)
        for col in ["cpc", "roas"]:
            df[f"log_{col}"] = np.log1p(df[col])
        return df


def codificador(binarias, numericas, categoricas):
    """Binárias sem escala (o coeficiente vira "efeito de ter o atributo"), numéricas padronizadas, categóricas em one-hot."""
    return ColumnTransformer([
        ("bin", SimpleImputer(strategy="most_frequent", add_indicator=True), binarias),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
    ], verbose_feature_names_out=False)


def codificador_pre_veiculacao():
    return codificador(CRIATIVO_BINARIAS + CONTEXTO_BINARIAS, CRIATIVO_NUMERICAS,
                       CRIATIVO_CATEGORICAS + CONTEXTO_CATEGORICAS)


def codificador_pos_veiculacao():
    return codificador(CRIATIVO_BINARIAS + CONTEXTO_BINARIAS + METRICAS_INDICADORAS,
                       CRIATIVO_NUMERICAS + METRICAS_NUMERICAS,
                       CRIATIVO_CATEGORICAS + CONTEXTO_CATEGORICAS)
