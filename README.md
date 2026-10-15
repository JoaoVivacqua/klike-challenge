# Klike Data Science Challenge

Análise de 500 campanhas de vídeo veiculadas em Meta, TikTok e LinkedIn: exploração dos dados, um modelo que prevê o `klike_score` (qualidade do criativo, de 0 a 100) e um engine que sugere melhorias para cada campanha.

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
6. Relação entre pares de variáveis (ex: campanhas com hook têm CTR maior?)
7. Efeito de uma terceira variável (a relação se mantém dentro de cada plataforma?)
8. Registro dos achados

<details>
<summary><b>Por que seguir um roteiro?</b></summary>

Cada passo depende do anterior, não adianta medir a correlação entre duas variáveis (passo 6) se uma delas tem valores impossíveis (passo 4), nem confiar numa relação geral sem checar se ela se mantém dentro de cada plataforma (passo 7): uma relação observada no total pode sumir ou até se inverter dentro dos grupos. Seguir a mesma ordem evita pular verificações e torna a análise fácil de acompanhar.

</details>

### 1. Contexto

Cada linha é uma campanha: um vídeo veiculado em uma plataforma, com seus atributos e métricas de performance. O alvo é o `klike_score`. As 26 colunas se dividem em quatro grupos:

| Grupo | Colunas | O que representam |
|---|---|---|
| Identificação | `campaign_id`, `date` | Qual campanha e quando |
| Contexto da campanha | `platform`, `category`, `objective`, `target_audience_age`, `is_retargeting` | Decisões de mídia |
| Atributos do criativo | `video_duration_s`, `format`, `has_subtitle`, `has_cta`, `has_hook`, `has_face`, `text_density`, `music_voice_ratio` | Decisões de produção do vídeo |
| Métricas de performance | `impressions`, `clicks`, `ctr`, `cpc`, `spend`, `conversions`, `revenue`, `roas`, `avg_watch_time_s`, `engagement_rate` | Resultados depois que o anúncio rodou |

A pergunta central da EDA ("quais atributos do criativo se correlacionam com boas métricas de performance?") compara o terceiro grupo com o quarto.

### 2. Divisão treino/teste antes de explorar

Decisão: separar 20% das campanhas (100) como teste antes de qualquer análise, com `random_state=42` e estratificação por plataforma.

Por quê: o teste simula campanhas novas. Se ele influenciar as decisões da análise, ocorre um vazamento de dados: as escolhas ficam ajustadas ao próprio teste (um tipo de overfitting), e a nota final do modelo fica otimista demais.

| Plataforma | Base | Treino | Teste |
|---|---|---|---|
| Meta | 48,2% | 48,2% | 48,0% |
| TikTok | 35,2% | 35,2% | 35,0% |
| LinkedIn | 16,6% | 16,5% | 17,0% |

<details>
<summary><b>Por que dividir antes de explorar?</b></summary>

O conjunto de teste existe para medir como o modelo se sairia com campanhas que nunca viu. Se explorássemos os 500 registros antes de dividir, o teste influenciaria nossas decisões mesmo sem querer:

- a mediana usada para preencher valores faltantes incluiria as campanhas de teste;
- as features seriam escolhidas olhando também para o teste;
- os outliers a remover seriam decididos vendo como aparecem no teste.

O teste deixaria de ser "novo": as decisões ficariam ajustadas a ele, um tipo de overfitting ao conjunto de teste, e a nota final seria otimista. Esse cuidado é conhecido como evitar *data snooping*. Por isso, toda a análise usa apenas o treino (400 campanhas).

</details>

<details>
<summary><b>Por que estratificar por plataforma?</b></summary>

Num sorteio simples, cada campanha tem a mesma chance de cair no teste, sem olhar a plataforma. O LinkedIn tem só 83 campanhas (17% da base), então o teste deveria receber cerca de 17, mas por acaso poderia receber bem menos ou bem mais. Numa base pequena, isso deixaria a avaliação por plataforma instável. Na divisão estratificada, o sorteio é feito dentro de cada plataforma (20% de cada uma), e os dois conjuntos ficam com a mesma composição da base, como mostra a tabela acima.

A plataforma foi escolhida sem olhar os dados: o enunciado destaca que "o que funciona no TikTok pode não funcionar no LinkedIn", e o engine de recomendações precisa considerar a plataforma.

Estratificar não causa vazamento de dados, ele só acontece quando informação do conteúdo do teste chega ao treino, como nos exemplos da seção anterior. A estratificação só controla quantas campanhas de cada plataforma vão para cada lado; nenhum atributo nem `klike_score` das campanhas de teste é usado. A proporção de plataformas no treino seria praticamente a mesma com um sorteio simples. A estratificação só remove a variação causada pelo acaso.

</details>

<details>
<summary><b>Por que 80/20 e random_state=42?</b></summary>

- 80/20 é um equilíbrio usual entre ter dados suficientes para aprender (400 campanhas) e uma avaliação de tamanho razoável (100).
- `random_state=42` fixa o sorteio, para que qualquer pessoa que rode o notebook obtenha exatamente a mesma divisão. O valor em si é arbitrário.

</details>

### 3. Formato e tipos

O que vimos: a maioria das colunas foi lida com o tipo certo, e as categorias batem com o dicionário de dados, sem variações de escrita. Três colunas precisam de ajuste:

- `date` foi lida como texto, não como data.
- `has_subtitle` foi lida como texto ("True"/"False"), porque tem 36 valores vazios no treino.
- `text_density` (low < medium < high) e `target_audience_age` (18-24 < 25-34 < 35-44 < 45+) têm ordem natural, mas foram lidas como texto sem ordem.

Decisão: corrigir esses tipos no pré-processamento, e não durante a análise.

Por quê: assim a mesma correção vale automaticamente para qualquer base nova (ex: uma campanha que o engine de recomendações vai avaliar), em vez de ficar presa a este notebook.

<details>
<summary><b>Como as categorias foram verificadas? E se surgirem novas plataformas no futuro?</b></summary>

O notebook compara as categorias encontradas no treino com as definidas no dicionário de dados do desafio (ex: as 3 plataformas, os 5 objetivos). Essa verificação pega erros de escrita, como "meta" em vez de "Meta", que fariam a mesma plataforma ser tratada como duas diferentes. No treino, todas as colunas bateram com o dicionário.

Uma categoria nova (ex: uma campanha do YouTube) não é tratada como erro: ela é apenas sinalizada como "fora do dicionário". Novas plataformas e categorias são esperadas conforme o produto cresce, e o aviso existe porque o modelo não terá aprendido nada sobre elas, então as previsões para esses casos seriam pouco confiáveis. Pelo mesmo motivo, o pré-processamento será construído para aceitar categorias novas sem quebrar.

</details>

### 4. Integridade

#### 4.1 Valores faltantes

O que vimos:

- 7 colunas têm vazios, entre 5% e 9% cada: `video_duration_s`, `has_subtitle`, `music_voice_ratio`, `cpc`, `revenue`, `avg_watch_time_s` e `engagement_rate`.
- 62,5% das campanhas estão completas, e quase todas as outras têm um único campo vazio.
- As taxas de vazios são parecidas entre Meta, TikTok e LinkedIn.
- Em 6 das 7 colunas, as campanhas com vazio não diferem das demais em nenhuma característica: os valores parecem faltar ao acaso (MCAR).
- A exceção é `avg_watch_time_s`: metade dos seus vazios (11 de 22) está nos 20% de campanhas com menos impressões. Uma explicação plausível é que, com poucas visualizações, a plataforma não reporta o tempo médio assistido. Isso indica vazios ligados a outra coluna (MAR).

Decisão: o tratamento dos vazios fica para o pré-processamento (Parte 2), e não para a análise.

Por quê: pelo mesmo motivo das correções de tipo, o tratamento precisa valer para qualquer base nova. O diagnóstico daqui orienta a escolha: para vazios ao acaso, preencher com um valor típico (ex: a mediana) é razoável; para `avg_watch_time_s`, o preenchimento deve considerar o tamanho da campanha, já que os vazios se concentram nas pequenas. Antes de decidir, o passo 4.3 verifica se `cpc` e `revenue` podem ser recalculados a partir de outras colunas (`spend / clicks` e `roas × spend`); se puderem, os vazios delas se resolvem com uma conta exata, sem estimativa.

<details>
<summary><b>Como saber se um valor falta ao acaso?</b></summary>

Um valor pode faltar por três motivos, conhecidos pelas siglas em inglês MCAR, MAR e MNAR. Cada um pede um tratamento diferente:

- MCAR (*Missing Completely At Random*, ao acaso): o vazio não tem relação com nada (ex: uma falha pontual de registro). Preencher com um valor típico funciona bem.
- MAR (*Missing At Random*, ligado a outra coluna): o vazio aparece mais em certo tipo de campanha (ex: campanhas pequenas). O preenchimento precisa levar essa coluna em conta, senão distorce os dados.
- MNAR (*Missing Not At Random*, ligado ao próprio valor): o vazio depende do valor que faltou (ex: um tempo assistido tão baixo que não foi registrado). Esse caso não pode ser verificado diretamente, porque o valor não existe.

O nome MAR confunde: apesar de "At Random", o vazio não é aleatório, e sim explicado por colunas que temos. Para distinguir MCAR de MAR, o notebook compara, para cada coluna com vazios, as campanhas com e sem o vazio em todas as outras características. Para as categóricas (ex: plataforma) usamos o teste qui-quadrado, que verifica se as proporções diferem; para as numéricas (ex: impressões), o teste de Mann-Whitney, que compara as distribuições sem exigir que sigam uma curva normal. Se os vazios fossem MCAR, os dois grupos seriam parecidos. Esse teste não descarta MNAR: um vazio ligado ao próprio valor pode passar despercebido, porque o valor que faltou não está disponível para comparação.

</details>

<details>
<summary><b>Por que 4 diferenças em 117 testes não indicam um padrão?</b></summary>

Cada teste usa o limite usual de p < 0,05, o que significa aceitar 5% de chance de apontar uma diferença que não existe. Com 117 comparações, cerca de 6 dariam "diferença" por puro acaso, mesmo que todos os vazios fossem aleatórios. Encontramos 4, então o resultado geral é compatível com o acaso.

`avg_watch_time_s` foi tratado como exceção porque não depende só do p-valor: as campanhas sem o valor têm cerca de três vezes menos impressões e gasto, o padrão aparece de forma nítida ao dividir as campanhas em faixas de tamanho, e existe uma explicação plausível para ele. Como o padrão foi encontrado depois de procurar, ele é tratado como hipótese provável, e não como certeza.

</details>
