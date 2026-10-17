import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import clone

from src.modelo import ALVO, CAMINHO_DADOS, carregar_treino_teste, modelo_pos_veiculacao, modelo_pre_veiculacao
from src.preprocessamento import TETO_CTR

MIN_CAMPANHAS = 15  # mínimo por grupo para estimar um efeito só da plataforma ou da faixa etária da campanha
MIN_GANHO = 0.5  # pontos de score


ACOES_CRIATIVO = {
    "hook": ("has_hook", 1.0, "Adicionar um hook nos 3 primeiros segundos"),
    "rosto humano": ("has_face", 1.0, "Incluir um rosto humano no vídeo"),
    "legenda": ("has_subtitle", 1.0, "Adicionar legendas"),
    "call-to-action": ("has_cta", 1.0, "Incluir um call-to-action"),
}

OBJETIVOS_DE_RESPOSTA = ["conversions", "traffic", "app_install"]

# comparações que o engine pode citar: (coluna, valor, referência, métrica, contexto)
CONTRASTES = (
    [(coluna, valor, 0.0, metrica, "target_audience_age" if (coluna, metrica) == ("has_face", "engagement_rate") else "platform")
     for coluna, valor, _ in ACOES_CRIATIVO.values() for metrica in ["ctr", "engagement_rate"]]
    + [("text_density", "medium", de, metrica, "platform") for de in ["high", "low"] for metrica in ["ctr", "engagement_rate"]]
    + [("is_retargeting", 1.0, 0.0, metrica, "platform") for metrica in ["conversion_rate", "roas"]]
)


def _formatar_pct(valor, casas=1):
    return f"{valor * 100:.{casas}f}%".replace(".", ",")

def _formatar_numero(valor, casas=1, sinal=False):
    return f"{valor:{'+' if sinal else ''}.{casas}f}".replace(".", ",")

def _taxa_para_log(valor):
    return np.log1p(valor * 100)


def _juntar(acao, detalhes):
    """Ação seguida dos detalhes, em frases separadas."""
    return f"{acao}: {detalhes[0]}" + "".join(f". {d[0].upper()}{d[1:]}" for d in detalhes[1:]) + "."


def _holm(p):
    ordem = np.argsort(p)
    ajustado = np.empty(len(p))
    ajustado[ordem] = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * np.asarray(p)[ordem]), 1)
    return ajustado


METRICAS_PERFORMANCE = [
    ("ctr", "log_ctr", _taxa_para_log, "o CTR",
     "testar em A/B variações da abertura e da primeira cena do vídeo", _formatar_pct, False),
    ("watch_ratio", "watch_ratio", lambda v: v, "a retenção (fração do vídeo assistida)",
     "revisar o ritmo do vídeo para segurar a audiência até o fim (ex: cortar trechos lentos)", lambda v: _formatar_pct(v, 0), False),
    ("engagement_rate", "log_engagement_rate", _taxa_para_log, "a taxa de engajamento",
     "incluir um convite explícito à interação (comentar, compartilhar)", lambda v: _formatar_pct(v, 2), False),
    ("conversion_rate", "log_conversion_rate", _taxa_para_log, "a taxa de conversão",
     "alinhar a promessa do vídeo com a oferta da página de destino", _formatar_pct, True),
    ("roas", "log_roas", np.log1p, "o ROAS",
     "revisar oferta e segmentação para aumentar a receita por real investido", lambda v: _formatar_numero(v, 2), True),
]


class EngineRecomendacoes:
    def __init__(self, n_bootstrap=200, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        self.modelo_pre_ = modelo_pre_veiculacao().fit(X, y)
        self.modelo_pos_ = modelo_pos_veiculacao().fit(X, y)

        Z, ridge = self.modelo_pre_[:-1].transform(X), self.modelo_pre_[-1]
        rng = np.random.default_rng(self.random_state)
        amostras = (rng.integers(0, len(Z), len(Z)) for _ in range(self.n_bootstrap))
        self.coefs_bootstrap_ = np.array([clone(ridge).fit(Z[i], y.to_numpy()[i]).coef_ for i in amostras])
        self.base_ = self.modelo_pre_[0].transform(X)
        metricas = self.base_.groupby("platform")[[m for m, *_ in METRICAS_PERFORMANCE]]
        self.referencias_ = {"da mediana": metricas.median(), "do quartil superior": metricas.quantile(0.75),
                             "do nível dos 10% melhores": metricas.quantile(0.9)}
        self.efeitos_ = self._efeitos_significativos()
        return self

    def _efeitos_significativos(self):
        """Efeitos nas métricas, em cada grupo de contexto e no total, mantidos só se significativos após Holm."""
        testes = []
        for coluna, valor, referencia, metrica, contexto in CONTRASTES:
            grupos = [(nivel, base) for nivel, base in self.base_.groupby(contexto)] + [(None, self.base_)]
            for nivel, base in grupos:
                com = base.loc[base[coluna] == valor, metrica].dropna()
                sem = base.loc[base[coluna] == referencia, metrica].dropna()
                if min(len(com), len(sem)) >= MIN_CAMPANHAS and sem.median() > 0:
                    testes.append(((coluna, valor, referencia, metrica, nivel), com.median() / sem.median() - 1,
                                   mannwhitneyu(com, sem).pvalue))
        chaves, efeitos, p = zip(*testes)
        return {chave: efeito for chave, efeito, p_holm in zip(chaves, efeitos, _holm(p)) if p_holm < 0.05}

    def _efeito_contextual(self, campanha, coluna, valor, referencia, metrica, contexto="platform"):
        nivel = campanha[contexto]
        if (coluna, valor, referencia, metrica, nivel) in self.efeitos_:
            onde = f"no {nivel}" if contexto == "platform" else f"no público de {nivel} anos"
            return self.efeitos_[coluna, valor, referencia, metrica, nivel], onde
        return self.efeitos_.get((coluna, valor, referencia, metrica, None)), "nas três plataformas juntas"

    @staticmethod
    def _texto_ganho(score_atual, ganho, inferior, superior):
        return (f"pode aumentar o klike_score em cerca de {_formatar_numero(ganho)} pontos (previsão de "
                f"{_formatar_numero(score_atual, 0)} para {_formatar_numero(score_atual + ganho, 0)}, IC 95%: "
                f"{_formatar_numero(inferior, sinal=True)} a {_formatar_numero(superior, sinal=True)})")

    def _ganhos_score(self, original, alteradas):
        Z = self.modelo_pre_[:-1].transform(pd.concat([original] + alteradas, ignore_index=True))
        previsto = self.modelo_pre_[-1].predict(Z)
        ganhos_bootstrap = (Z[1:] - Z[0]) @ self.coefs_bootstrap_.T
        return previsto[0], previsto[1:] - previsto[0], np.percentile(ganhos_bootstrap, [2.5, 97.5], axis=1).T

    def _recomendacoes_criativo(self, original, limpa):
        campanha = limpa.iloc[0]
        acoes = {nome: acao for nome, acao in ACOES_CRIATIVO.items() if campanha[acao[0]] != acao[1]}
        if campanha["text_density"] != "medium":
            de = {"high": "alto", "low": "baixo"}[campanha["text_density"]]
            verbo = "Reduzir" if de == "alto" else "Aumentar"
            acoes["texto"] = ("text_density", "medium", f"{verbo} a quantidade de texto na tela de {de} para médio")
        if not acoes:
            return []

        alteradas = [original.assign(**{coluna: valor}) for coluna, valor, _ in acoes.values()]
        score_atual, ganhos, intervalos = self._ganhos_score(original, alteradas)
        recomendacoes = []
        for (nome, (coluna, valor, acao)), ganho, (inferior, superior) in zip(acoes.items(), ganhos, intervalos):
            if ganho <= 0:
                continue
            referencia = campanha[coluna] if coluna == "text_density" else 0.0
            detalhes = [self._texto_ganho(score_atual, ganho, inferior, superior)]

            efeito_ctr, onde = self._efeito_contextual(campanha, coluna, valor, referencia, "ctr")
            if efeito_ctr is not None and efeito_ctr > 0.05 and pd.notna(campanha["ctr"]):
                ctr_novo = min(campanha["ctr"] * (1 + efeito_ctr), TETO_CTR)
                teto = ", o teto observado nos dados" if ctr_novo == TETO_CTR else ""
                detalhes.append(f"{onde}, campanhas assim têm CTR mediano {_formatar_pct(efeito_ctr, 0)} maior "
                                f"(o desta passaria de {_formatar_pct(campanha['ctr'])} para cerca de {_formatar_pct(ctr_novo)}{teto})")

            contexto = "target_audience_age" if nome == "rosto humano" else "platform"
            efeito_eng, onde = self._efeito_contextual(campanha, coluna, valor, referencia, "engagement_rate", contexto)
            if efeito_eng is not None and efeito_eng > 0.05:
                detalhes.append(f"{onde}, a taxa de engajamento mediana é {_formatar_pct(efeito_eng, 0)} maior")
            recomendacoes.append({"tipo": "criativo", "ganho_score": ganho, "recomendacao": _juntar(acao, detalhes)})
        return recomendacoes

    def _recomendacoes_performance(self, limpa):
        campanha = limpa.iloc[0]
        resposta = campanha["objective"] in OBJETIVOS_DE_RESPOSTA
        modelo = self.modelo_pos_[1:]
        atual = modelo.predict(limpa)[0]
        recomendacoes = []
        for metrica, feature, transformar, nome, acao, formatar, so_resposta in METRICAS_PERFORMANCE:
            valor = campanha[metrica]
            if pd.isna(valor) or (so_resposta and not resposta):
                continue
            referencias = [(rotulo, ref.loc[campanha["platform"], metrica]) for rotulo, ref in self.referencias_.items()]
            alvo = next(((rotulo, ref) for rotulo, ref in referencias if valor < ref), None)
            if alvo is None:
                continue
            rotulo, referencia = alvo
            alterada = limpa.copy()
            alterada[feature] = transformar(referencia)
            ganho = modelo.predict(alterada)[0] - atual
            if ganho <= 0:
                continue
            recomendacoes.append({"tipo": "performance", "ganho_score": ganho, "recomendacao": (
                f"{nome[0].upper() + nome[1:]} desta campanha ({formatar(valor)}) está abaixo {rotulo} das campanhas do "
                f"{campanha['platform']} ({formatar(referencia)}). Sugestão: {acao}. Chegar a esse nível elevaria o "
                f"klike_score em cerca de {_formatar_numero(ganho)} pontos.")})
        return recomendacoes

    def _recomendacoes_midia(self, original, limpa):
        campanha = limpa.iloc[0]
        if campanha["is_retargeting"] == 1 or campanha["objective"] not in OBJETIVOS_DE_RESPOSTA:
            return []
        score_atual, (ganho,), ((inferior, superior),) = self._ganhos_score(original, [original.assign(is_retargeting=True)])
        if ganho <= 0:
            return []
        detalhes = [self._texto_ganho(score_atual, ganho, inferior, superior)]
        conversao, onde = self._efeito_contextual(campanha, "is_retargeting", 1.0, 0.0, "conversion_rate")
        roas, _ = self._efeito_contextual(campanha, "is_retargeting", 1.0, 0.0, "roas")
        if conversao is not None:
            efeito = f"{onde}, campanhas de retargeting têm taxa de conversão mediana {_formatar_pct(conversao, 0)} maior"
            if roas is not None and campanha["roas_nao_se_aplica"] == 0:
                efeito += f" e ROAS {_formatar_pct(roas, 0)} maior"
            detalhes.append(efeito)
        return [{"tipo": "mídia", "ganho_score": ganho,
                 "recomendacao": _juntar("Direcionar parte do orçamento para retargeting", detalhes)}]

    def _recomendacao_manter(self, original, limpa):
        campanha = limpa.iloc[0]
        presentes = {nome: (col, 0.0) for nome, (col, valor, _) in ACOES_CRIATIVO.items() if campanha[col] == valor}
        if campanha["text_density"] == "medium":
            presentes["texto médio"] = ("text_density", "high")
        if not presentes:
            return []
        _, ganhos, _ = self._ganhos_score(original, [original.assign(**{c: v}) for c, v in presentes.values()])
        perdas = sorted(zip(presentes, -ganhos), key=lambda par: -par[1])
        lista = ", ".join(f"{nome} ({_formatar_numero(perda)} pontos)" for nome, perda in perdas)
        return [{"tipo": "manter", "ganho_score": 0.0, "recomendacao": (
            f"Manter nas próximas versões os atributos que já sustentam o score. Removê-los custaria cerca de: {lista}.")}]

    def recomendar(self, campanha, maximo=5):
        original = pd.DataFrame([dict(campanha)]).drop(columns=[ALVO], errors="ignore")
        limpa = self.modelo_pre_[0].transform(original)
        performance = self._recomendacoes_performance(limpa)
        principais = (self._recomendacoes_criativo(original, limpa) + self._recomendacoes_midia(original, limpa)
                      + [r for r in performance if r["ganho_score"] >= MIN_GANHO])
        recomendacoes = sorted(principais, key=lambda r: -r["ganho_score"])
        
        # para campanhas já fortes completamos com ganhos menores e com o que manter
        if len(recomendacoes) < 3:
            recomendacoes += sorted([r for r in performance if r["ganho_score"] < MIN_GANHO], key=lambda r: -r["ganho_score"])
        if len(recomendacoes) < 3:
            recomendacoes += self._recomendacao_manter(original, limpa)
        tabela = pd.DataFrame(recomendacoes, columns=["tipo", "ganho_score", "recomendacao"]).head(maximo)
        tabela.insert(0, "prioridade", range(1, len(tabela) + 1))
        return tabela.reset_index(drop=True)


def main(campaign_id):
    df = pd.read_csv(CAMINHO_DADOS)
    if campaign_id not in set(df["campaign_id"]):
        sys.exit(f"Campanha {campaign_id} não encontrada.")
    X_train, y_train, _, _ = carregar_treino_teste()
    engine = EngineRecomendacoes().fit(X_train, y_train)
    campanha = df.loc[df["campaign_id"] == campaign_id].iloc[0]
    print(f"\n{campaign_id} | {campanha['platform']} | {campanha['category']} | objetivo: {campanha['objective']} "
          f"| klike_score real: {campanha[ALVO]}\n")
    print("Os ganhos são estimados um a um e não devem ser somados.\n")
    for _, linha in engine.recomendar(campanha).iterrows():
        print(f"{linha['prioridade']}. [{linha['tipo']}] {linha['recomendacao']}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "KLK-0001")
