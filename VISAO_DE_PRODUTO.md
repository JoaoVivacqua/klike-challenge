# Parte 4 — Visão de Produto

Respostas às três perguntas abertas do desafio. Cada resposta parte do que os dados mostraram nas Partes 1 a 3 ([README](README.md)).

## 1. Que features eu extrairia do vídeo original

Resposta curta: Eu mediria os atributos que já mostraram efeito no score (hook, texto na tela, rosto), em vez de só marcar se o vídeo tem ou não tem cada um.

Por quê: O modelo atual não decora os dados (o erro no treino e no teste é o mesmo), e modelos mais flexíveis não melhoraram o resultado (Parte 2). O erro que sobra vem da falta de informação, e não do algoritmo. Hoje, dois vídeos com "hook: sim" são iguais para o modelo, mesmo que um prenda a atenção no primeiro segundo e o outro só no terceiro.

Em ordem de prioridade:

1. Hook medido, e não marcado: Tempo até a primeira fala, o primeiro rosto e o primeiro corte, e quanto movimento há nos 3 primeiros segundos. O hook é o atributo que mais pesa no score (cerca de 17 pontos).

2. Texto na tela, medido automaticamente: Quantidade de palavras, tamanho e contraste da fonte, e se o texto cai nas áreas cobertas pelos botões e legendas de cada plataforma. A densidade de texto é o segundo maior efeito no score e hoje só existe em três níveis (baixo, médio, alto).

3. Rosto: Tempo em tela, em que segundo aparece, close ou plano aberto, olhar para a câmera.

4. Áudio: Se há fala nos 3 primeiros segundos e o que é dito (transcrição), como menção a oferta, preço ou problema do cliente. A proporção entre música e voz não teve efeito, o que sugere que importa o conteúdo da fala, e não a proporção.

5. Ritmo e marca: Cortes por segundo e em que momento o produto e a marca aparecem.

6. Estilo do vídeo: Classificar como o vídeo é feito (depoimento, demonstração do produto, conteúdo gravado por um usuário) e encontrar, entre os criativos de score alto, os mais parecidos com o da campanha, para usar como referência. O dataset tem o objetivo da campanha (ex: conversão), mas não o estilo: um vídeo de conversão pode ser um depoimento ou uma demonstração.

Junto com o vídeo, eu buscaria nas plataformas a curva de retenção segundo a segundo, ela mostra em que ponto a audiência abandona o vídeo e permitiria recomendar cortes em trechos específicos.

O ganho não ficaria só no erro do modelo, as recomendações ficam mais específicas: em vez de "adicionar um hook", algo como "o seu hook começa em 2,5 segundos, nos vídeos de score alto, começa antes de 1 segundo".

## 2. Como eu colocaria o engine em produção

Pelo que o site do Klike mostra, o cliente envia o criativo para análise antes de publicar, conecta a conta de anúncios e recebe de volta o resultado real das campanhas. Há também acesso por API no plano Enterprise e um conector para assistentes de IA, o engine se encaixaria nesses fluxos, sem criar um fluxo novo.

### Dois momentos no produto

- No envio do criativo, antes de ir ao ar: O modelo pré-veiculação gera as recomendações de criativo, o cliente vê o score previsto e o que mudar antes de gastar com mídia.

- Depois que a campanha começa a rodar, com as métricas da conta conectada: O modelo pós-veiculação gera as recomendações de performance, comparando a campanha com a referência da plataforma.

Os dois modelos do projeto foram separados exatamente por isso: as métricas de performance só existem no segundo momento.

### Arquitetura

```
envio do vídeo ──> fila ──> extração de features do vídeo ──────┐
conta conectada ──> coleta diária das métricas das plataformas ─┤
                                                                v
                         serviço de recomendações <── modelo treinado
                                     │
              tela do Klike · API Enterprise · conector para assistentes
```

- Serviço de recomendações: Uma API que recebe a campanha e devolve a lista priorizada, a partir do pipeline treinado (limpeza, codificação e modelo), que o serviço só carrega para fazer previsões. As referências por plataforma e os efeitos significativos já são calculados no treino, como o fit do engine faz hoje, então cada recomendação sai em milissegundos.

- Extração de features do vídeo: É a parte pesada. Roda em segundo plano, numa fila, com máquinas que aumentam ou diminuem conforme o volume de envios.

- Métricas: Coletadas uma vez por dia pelas APIs do Meta, do TikTok e do LinkedIn, as regras de limpeza da EDA (contas que não fecham, valores no teto) viram checagens automáticas de qualidade antes de os dados chegarem ao modelo.

- Escala: A previsão é barata, e o custo está no processamento de vídeo, que escala adicionando máquinas à fila. A mesma versão do modelo atende a tela do produto, a API e o conector.

### Monitoramento e evolução

- Dados: Alertas para categorias novas (o pipeline já aceita uma plataforma ou categoria desconhecida sem quebrar, mas a previsão para ela é menos confiável), para mudanças na distribuição das features e para erros de registro.

- Modelo: Quando o score real de uma campanha chega, comparar com o previsto, retreinar periodicamente com as campanhas novas e só publicar uma versão nova se ela errar menos na validação.

- Uso: Registrar se o cliente aplicou cada recomendação e o que aconteceu com a campanha depois.

## 3. O que eu faria com mais tempo

Em ordem de prioridade:

1. Medir o efeito causal das recomendações, os efeitos de hoje são associações: campanhas com hook têm score maior, mas o dataset não prova que adicionar um hook causa o aumento. O Klike já oferece teste A/B. Comparar versões do mesmo vídeo com e sem a recomendação aplicada mostraria quanto ela vale de verdade, e esses resultados poderiam recalibrar os ganhos que o engine estima.

2. Considerar o custo de cada ação. Adicionar legenda é barato, regravar com um rosto humano é caro, com uma estimativa de custo por ação, o engine poderia recomendar a combinação de mudanças com o melhor retorno pelo esforço, em vez de ações isoladas, cujos ganhos hoje não podem ser somados.

3. Modelar o CTR e o engajamento diretamente. O engine quantifica o ganho no score com o modelo, mas o efeito nas métricas vem de comparações de medianas. Modelos próprios para essas métricas dariam estimativas que levam em conta os outros atributos da campanha, com intervalo de confiança.

4. Mais dados do LinkedIn. São só 66 campanhas no treino, e vários efeitos não chegam a ser significativos lá, com mais dados, um modelo hierárquico, que compartilha informação entre as plataformas mas permite diferenças, mostraria se o LinkedIn realmente se comporta como as outras.

5. Testes automatizados do engine (toda campanha recebe ao menos 3 recomendações, os ganhos têm o sinal esperado) e uma interface simples, para o time de marketing testar campanhas sem escrever código.
