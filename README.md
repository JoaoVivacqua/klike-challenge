# Klike Data Science Challenge

Análise de 500 criativos de vídeo veiculados em Meta, TikTok e LinkedIn: exploração dos dados, um modelo que prevê o `klike_score` (qualidade do criativo, de 0 a 100) e um engine que sugere melhorias para cada campanha.

## Resumo

- Antes de explorar, separamos 20% dos dados para teste, que só serão usados na avaliação final do modelo.

Os notebooks contêm o código e os resultados. Este README explica as decisões.

## Como rodar

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
python -m notebook notebooks/
```

## Estrutura

```
├── data/
│   └── klike_challenge_dataset.csv   # dataset original, sem alterações
├── notebooks/
│   └── 01_eda.ipynb                  # Parte 1: análise exploratória
├── requirements.txt
└── README.md
```

---

## Parte 1 — Análise Exploratória

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

A análise segue um roteiro fixo, do mais básico ao mais detalhado:

1. Contexto
2. Divisão treino/teste
3. Formato e tipos
4. Integridade (faltantes, duplicatas, valores impossíveis, tetos)
5. Distribuição de cada variável (valores comuns e raros, média vs. mediana)
6. Relação entre pares de variáveis (ex: criativos com hook têm CTR maior?)
7. Terceiras variáveis (a relação se mantém dentro de cada grupo?)
8. Registro dos achados

<details>
<summary>Ver detalhes: por que seguir um roteiro</summary>

Cada passo depende do anterior. Não adianta medir a correlação entre duas variáveis (passo 6) se uma delas tem valores impossíveis (passo 4), nem confiar numa relação geral sem checar se ela se mantém dentro de cada plataforma (passo 7): uma relação observada no total pode sumir ou até se inverter dentro dos grupos. Seguir a mesma ordem evita pular verificações e torna a análise fácil de acompanhar.
</details>

### 1. Contexto

Cada linha é um criativo de vídeo veiculado em uma campanha. O alvo é o `klike_score`. As 26 colunas se dividem em quatro grupos:

| Grupo | Colunas | O que representam |
|---|---|---|
| Identificação | `campaign_id`, `date` | Qual campanha e quando |
| Contexto da campanha | `platform`, `category`, `objective`, `target_audience_age`, `is_retargeting` | Decisões de mídia |
| Atributos do criativo | `video_duration_s`, `format`, `has_subtitle`, `has_cta`, `has_hook`, `has_face`, `text_density`, `music_voice_ratio` | Decisões de produção do vídeo |
| Métricas de performance | `impressions`, `clicks`, `ctr`, `cpc`, `spend`, `conversions`, `revenue`, `roas`, `avg_watch_time_s`, `engagement_rate` | Resultados depois que o anúncio rodou |

A pergunta central da EDA ("quais atributos do criativo se correlacionam com boas métricas de performance?") compara o terceiro grupo com o quarto.

### 2. Divisão treino/teste antes de explorar

Decisão: separar 20% das campanhas (100) como teste antes de qualquer análise, com `random_state=42` e estratificação por plataforma.

Por quê: o teste simula campanhas novas. Se ele influenciar as decisões da análise, a nota final do modelo fica otimista demais.

| Plataforma | Base | Treino | Teste |
|---|---|---|---|
| Meta | 48,2% | 48,2% | 48,0% |
| TikTok | 35,2% | 35,2% | 35,0% |
| LinkedIn | 16,6% | 16,5% | 17,0% |

<details>
<summary>Ver detalhes: por que dividir antes de explorar</summary>

O conjunto de teste existe para medir como o modelo se sairia com campanhas que nunca viu. Se explorássemos os 500 registros antes de dividir, o teste influenciaria nossas decisões mesmo sem querer:

- a mediana usada para preencher valores faltantes incluiria as campanhas de teste;
- as features seriam escolhidas olhando também para o teste;
- os outliers a remover seriam decididos vendo como aparecem no teste.

O teste deixaria de ser "novo", e a nota final seria otimista. É como um aluno que vê a prova antes de estudar: a nota sobe, mas não mede o que ele aprendeu. Esse cuidado é conhecido como evitar *data snooping*. Por isso, toda a análise usa apenas o treino (400 campanhas).
</details>

<details>
<summary>Ver detalhes: por que estratificar por plataforma (e por que isso não é vazamento)</summary>

Num sorteio simples, cada campanha tem a mesma chance de cair no teste, sem olhar a plataforma. O LinkedIn tem só 83 campanhas (17% da base), então o teste deveria receber cerca de 17, mas por acaso poderia receber bem menos ou bem mais. Numa base pequena, isso deixaria a avaliação por plataforma instável. Na divisão estratificada, o sorteio é feito dentro de cada plataforma (20% de cada uma), e os dois conjuntos ficam com a mesma composição da base, como mostra a tabela acima.

A plataforma foi escolhida sem olhar os dados: o enunciado destaca que "o que funciona no TikTok pode não funcionar no LinkedIn", e o engine de recomendações precisa considerar a plataforma.

Estratificar não causa vazamento de dados. Vazamento acontece quando informação do conteúdo do teste chega ao treino, como nos exemplos da seção anterior. A estratificação só controla quantas campanhas de cada plataforma vão para cada lado; nenhum atributo nem `klike_score` das campanhas de teste é usado. A proporção de plataformas no treino seria praticamente a mesma com um sorteio simples. A estratificação só remove a variação causada pelo acaso.
</details>

<details>
<summary>Ver detalhes: por que 80/20 e random_state=42</summary>

- 80/20 é um equilíbrio usual entre ter dados suficientes para aprender (400 campanhas) e uma avaliação de tamanho razoável (100).
- `random_state=42` fixa o sorteio, para que qualquer pessoa que rode o notebook obtenha exatamente a mesma divisão. O valor em si é arbitrário.
</details>
