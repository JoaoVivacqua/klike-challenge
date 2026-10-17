import numpy as np
import pandas as pd

TOLERANCIA = 0.2

def medianas_por_plataforma(df):
    """Medianas de ctr e cpc por plataforma. Calcular só no treino."""
    return df.groupby("platform")[["ctr", "cpc"]].median()


def _conta_quebrada(registrado, calculado):
    return (registrado / calculado - 1).abs() > TOLERANCIA


def _mais_plausivel(candidato_a, candidato_b, referencia):
    dist_a = np.abs(np.log(candidato_a / referencia))
    dist_b = np.abs(np.log(candidato_b / referencia))
    return dist_a <= dist_b


def corrigir_inconsistencias(df, medianas):
    indice = df.index
    df = df.reset_index(drop=True)
    df["coluna_corrigida"] = None
    ref_ctr = df["platform"].map(medianas["ctr"])
    ref_cpc = df["platform"].map(medianas["cpc"])

    ctr_calculado = df["clicks"] / df["impressions"]
    quebrada = _conta_quebrada(df["ctr"], ctr_calculado)
    erro_impressions = quebrada & _mais_plausivel(df["ctr"], ctr_calculado, ref_ctr)
    erro_ctr = quebrada & ~erro_impressions
    df.loc[erro_impressions, "impressions"] = (df["clicks"] / df["ctr"]).round().astype("int64")
    df.loc[erro_ctr, "ctr"] = ctr_calculado.round(4)
    df.loc[erro_impressions, "coluna_corrigida"] = "impressions"
    df.loc[erro_ctr, "coluna_corrigida"] = "ctr"

    cpc_calculado = df["spend"] / df["clicks"]
    quebrada = _conta_quebrada(df["cpc"], cpc_calculado)
    erro_spend = quebrada & _mais_plausivel(df["cpc"], cpc_calculado, ref_cpc)
    erro_cpc = quebrada & ~erro_spend
    df.loc[erro_spend, "spend"] = (df["cpc"] * df["clicks"]).round(2)
    df.loc[erro_cpc, "cpc"] = cpc_calculado.round(2)
    df.loc[erro_spend, "coluna_corrigida"] = "spend"
    df.loc[erro_cpc, "coluna_corrigida"] = "cpc"

    roas_quebrado = _conta_quebrada(df["roas"], df["revenue"] / df["spend"]) & (df["revenue"] > 0)
    df.loc[roas_quebrado, "coluna_corrigida"] = "roas não fecha (não corrigido)"
    df.index = indice
    return df
