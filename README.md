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
├── src/
│   └── limpeza.py                    # correção de inconsistências, usada na EDA e no pré-processamento
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
- Comparando as campanhas com e sem vazio em cada uma das outras variáveis, nenhuma diferença continua significativa após corrigir para comparações múltiplas: não há evidência contra MCAR.
- A exceção provável é `avg_watch_time_s`: metade dos seus vazios (11 de 22) está nos 20% de campanhas com menos impressões, o que indica MAR. Uma explicação plausível é que, com poucas visualizações, a plataforma não reporta o tempo médio assistido.

Decisão: o tratamento dos vazios fica para o pré-processamento (Parte 2), e não para a análise.

Por quê: pelo mesmo motivo das correções de tipo, o tratamento precisa valer para qualquer base nova. O diagnóstico daqui orienta a escolha: para vazios ao acaso, preencher com um valor típico (ex: a mediana) é razoável; para `avg_watch_time_s`, o preenchimento deve considerar o tamanho da campanha, já que os vazios se concentram nas pequenas. Os vazios de `cpc` e `revenue` não precisam de estimativa: o passo 4.3 mostra que todos podem ser recalculados por `spend / clicks` e `roas × spend`.

<details>
<summary><b>Como foi testado o mecanismo dos faltantes?</b></summary>

Para cada coluna com vazios, comparamos as campanhas com e sem o vazio nas demais variáveis: qui-quadrado nas categóricas e Mann-Whitney nas numéricas. Usamos Mann-Whitney em vez de comparar médias porque impressões, gasto e ROAS têm cauda longa (mesmo depois de corrigir os erros de registro do passo 4.3, a maior campanha tem cerca de 67 vezes as impressões da mediana), e uma única campanha extrema dominaria a média de um grupo.

Foram 117 comparações, então os p-valores foram corrigidos pelo método de Holm. Sem a correção, 4 comparações têm p < 0,05, menos que as cerca de 6 esperadas só por acaso; com a correção, nenhuma é significativa.

O teste só distingue MCAR de MAR. MNAR não pode ser verificado sem o valor ausente.

</details>

<details>
<summary><b>Por que avg_watch_time_s é tratado como exceção, se não passou na correção?</b></summary>

A correção de Holm controla os falsos positivos de uma busca ampla, mas também reduz o poder de detectar efeitos reais com poucos casos (22 vazios). Três evidências se somam: os menores p-valores da busca (0,002 para impressões e 0,003 para gasto), medianas de impressões e gasto cerca de três vezes menores nas campanhas com vazio e uma concentração nítida na faixa das menores campanhas. Existe também um mecanismo plausível. Como o padrão foi encontrado depois de procurar, ele é tratado como hipótese provável de MAR, e não como certeza. MNAR também é plausível, se um tempo assistido muito baixo simplesmente não for registrado. Nos dois casos, uma coluna indicadora de falta preservaria essa informação para o modelo.

</details>

#### 4.2 Duplicatas

O que vimos: nenhuma duplicata em quatro níveis de verificação: linhas idênticas, `campaign_id` repetido, linhas idênticas ignorando o `campaign_id` e campanhas com o mesmo contexto e o mesmo vídeo (plataforma, categoria, objetivo, público, retargeting e atributos do criativo). Também não há nenhuma campanha presente no treino e no teste ao mesmo tempo.

Decisão: nenhuma linha removida. A checagem de sobreposição entre treino e teste é a única do EDA que usa o teste.

Por quê: uma campanha repetida nos dois conjuntos seria vazamento de dados, porque o modelo seria avaliado em um exemplo já visto no treino. A checagem compara apenas se as linhas são iguais, sem calcular estatísticas nem olhar o `klike_score` do teste, então não influencia nenhuma decisão de análise. Os níveis mais amplos cobrem o caso de uma mesma campanha cadastrada duas vezes com IDs ou métricas diferentes, que um teste de linhas idênticas não pegaria.

#### 4.3 Valores impossíveis e inconsistentes

O que vimos:

- Nenhuma violação das regras que valem por definição: sem valores negativos, cliques nunca acima das impressões, conversões nunca acima dos cliques, tempo assistido nunca acima da duração do vídeo, taxas entre 0 e 1 e datas dentro de 2024.
- As métricas derivadas (`ctr = clicks / impressions`, `cpc = spend / clicks`, `roas = revenue / spend`) fecham com até 10% de diferença por arredondamento, exceto em 10 campanhas, em que a diferença passa de 70%. Não há casos intermediários.
- Cada uma dessas 10 campanhas tem exatamente uma coluna com erro de registro:

| Coluna com erro | Campanhas | Valor registrado | Valor recuperado |
|---|---|---|---|
| `impressions` | KLK-0148, 0232, 0258, 0413 | 12,6 a 14,1 milhões | 11.695 a 86.948 (`clicks / ctr`) |
| `cpc` | KLK-0100, 0240, 0416 | 20,00 a 26,05 | 0,39 a 0,98 (`spend / clicks`) |
| `spend` | KLK-0109, 0224, 0259 | 50.545 a 95.812 | 5,40 a 21.557 (`cpc × clicks`) |

- Receita zero não indica performance ruim: as 41 campanhas com receita zero e conversões registradas são todas de awareness ou engagement, e 47 das 49 campanhas de Branding têm receita zero. A receita dessas campanhas não é medida.
- Todos os vazios de `cpc` (20) e `revenue` (22) podem ser recalculados pelas outras colunas.

Decisões:

- Recuperar os valores das 10 campanhas pela própria conta, em vez de descartá-las ou marcá-las como faltantes. A regra fica em `src/limpeza.py` e é aplicada no pré-processamento, a qualquer base.
- Registrar a coluna alterada em `coluna_corrigida`, para auditoria. Ela não entra no modelo.
- Recalcular os vazios de `cpc` e `revenue` em vez de preenchê-los com a mediana.
- Tratar o `roas` de campanhas sem receita medida como "não se aplica", e não como zero. A forma exata fica para a Parte 2.

Por quê: o valor recuperado é exato, não uma estimativa, e descartar as 10 campanhas perderia 2,5% do treino sem necessidade. Manter os erros seria pior que descartar: são essas campanhas que criam a cauda absurda de impressões e gasto (14 milhões de impressões, gasto de 95 mil com 20 cliques), que distorceria qualquer média e o próprio modelo. Já `coluna_corrigida` não entra no modelo porque um erro de registro não diz nada sobre a qualidade do criativo; o valor dela é rastrear o que foi alterado.

<details>
<summary><b>Como a regra decide qual coluna está errada?</b></summary>

Quando `ctr = clicks / impressions` não fecha, o erro pode estar em `impressions` ou no próprio `ctr`; quando `cpc = spend / clicks` não fecha, em `spend` ou no `cpc`. A regra compara as duas versões possíveis da taxa (a registrada e a recalculada) com a mediana da plataforma no treino, em escala log, mantém a mais plausível e recalcula a outra coluna. Ex: em KLK-0259, o `cpc` registrado (0,27) é plausível para o TikTok, e o recalculado (4.790) não; logo o erro está no `spend`.

O `roas` não entra na regra e serve como validação independente: depois da correção, ele fecha nas 10 campanhas. Em KLK-0259, o gasto recuperado (5,40) reproduz exatamente o `roas` registrado (13,45). Quando só o `roas` não fecha, não há uma terceira conta para desempatar, e a campanha é apenas sinalizada; isso não ocorre no treino.

A tolerância de 20% fica no meio do vão entre o arredondamento (até ~8%) e os erros (acima de 70%), então a regra não é sensível ao valor exato escolhido. As medianas de referência são calculadas apenas no treino.
</details>

<details>
<summary><b>A correção muda as conclusões do passo 4.1?</b></summary>

Não. Os testes de faltantes do 4.1 foram feitos com os dados brutos, que incluem os 10 erros. Repetidos com os valores corrigidos, as mesmas 4 comparações têm p < 0,05 e nenhuma sobrevive à correção de Holm. O padrão de `avg_watch_time_s` se mantém (medianas de impressões de 9.045 com vazio e 25.057 sem).
</details>

#### 4.4 Tetos e pisos

O que vimos (no treino já corrigido no 4.3):

- `ctr` truncado no topo em 0,15: 42 campanhas (10,5%) estão exatamente nesse valor, contra 24 entre 0,13 e 0,15, e nenhuma acima. Os cliques foram ajustados junto (`clicks / impressions` também dá 0,15), então o CTR real dessas campanhas é desconhecido; só se sabe que é de pelo menos 15%.
- O teto do `ctr` não atinge campanhas ao acaso: as campanhas no teto têm `klike_score` mediano de 74,4, contra 59,3 nas demais (Mann-Whitney, p ≈ 2 × 10⁻⁷), e 74% delas usam hook, contra 49%.
- `engagement_rate` truncado em 0,10: 7 campanhas, todas do TikTok (5% da plataforma), sem diferença de `klike_score` em relação às demais (p ≈ 0,9).
- Pisos com efeito desprezível: `impressions` tem mínimo de exatamente 1.000 em 2 campanhas, cujo `ctr` indica 983 e 990; `video_duration_s` tem um leve acúmulo em 5 s (16 campanhas, contra 10 em 6 s), quase todas do TikTok.

Decisões:

- Manter os valores no teto e adicionar, no pré-processamento, as colunas indicadoras `ctr_no_teto` e `engagement_no_teto`.
- As recomendações do engine não projetam CTR acima de 15%.
- Pisos de `impressions` e `video_duration_s`: apenas registrados.

Por quê: remover as campanhas no teto descartaria 10,5% do treino e, como o teto concentra as campanhas de maior `klike_score`, enviesaria o modelo para baixo. Um modelo para dados censurados resolveria o problema de forma mais completa, mas é desproporcional para 400 campanhas. A coluna indicadora informa ao modelo que o valor real é "pelo menos 0,15", e não exatamente 0,15. Para a análise, o teto tem uma consequência direta: ele achata a diferença entre as melhores campanhas, então a relação entre CTR e `klike_score` vai parecer mais fraca do que é. No engine, prever CTR acima de 15% seria extrapolar além do que os dados mostram.
