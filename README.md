# Repositório do artigo **Mudanças Linguísticas em Artigos de Conferências Brasileiras de Computação no Contexto da Inteligência Artificial Generativa**
---
## Pacote `src_oficial` — IA generativa nas publicações SBC: web scraping, dataset corrigido, métricas, Pangram e análises

Pacote para **reproduzir de ponta a ponta** os resultados apresentados no artigo **Mudanças Linguísticas em Artigos de Conferências Brasileiras de Computação no Contexto da Inteligência Artificial Generativa**: web scraping, dataset, scripts de geração das métricas, do Pangram, das análises estatísticas e dos gráficos e scripts de apoio.

---

## 1. Conteúdo do pacote

```
src_oficial/
├── README.md                     este documento
├── rodar_tudo.sh                 executa o pipeline completo na ordem certa
├── requirements.txt              versões exatas do ambiente em que os resultados foram gerados
├── .env.example                  modelo do .env com HF_TOKEN2 (só para o Pangram)
├── web_scraping/                 recupera os artigos do SBC Open Lib e os salva em um dataset
├── src/                          todos os scripts (seção 5)
├── dataset/                      Nosso dataset: JSONs estruturados por <conferência>/<ano>/ (seção 4)
└── resultados/                   saídas que obtivemos (CSVs e figuras)
```
## 2. Como reproduzir

**Ambiente usado:** Linux, Python 3.13, GPU NVIDIA RTX 4070 SUPER (12 GB), CUDA 13; pacotes em `requirements.txt` (beautifulsoup, torch 2.13, transformers 5.14, peft 0.21, bitsandbytes 0.50, spaCy 3.8.16 com `pt_core_news_sm` e `en_core_web_sm` 3.8.0, lexicalrichness 0.5.1, textstat 0.7.13, pyphen 0.18.1, langdetect 1.0.9, scikit-learn 1.8, scipy 1.17, pandas 2.3.3, matplotlib 3.10.8).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download pt_core_news_sm && python -m spacy download en_core_web_sm
cp .env.example .env      # editar: HF_TOKEN2=<token de conta com acesso aprovado aos modelos do Pangram>
./rodar_tudo.sh
```
- **Pangram:** exige GPU e um token do Hugging Face (`HF_TOKEN2`) de conta com acesso **aprovado** (modelos "gated") a `meta-llama/Llama-3.2-3B` e `pangram/editlens_Llama-3.2-3B`. O código usa sempre `HF_TOKEN2` (sobrescreve `HF_TOKEN`).
- **Tempos aproximados (RTX 4070 SUPER):** métricas ~40 min (corpo + resumo), Pangram no corpo ~40 min, no resumo ~2 min, todo o resto poucos minutos. As etapas de métricas e de Pangram são retomáveis (rodar de novo sem `--refaz`).
- Os docstrings dos scripts dizem `python3 -I`; use `python3` simples se seus pacotes estiverem fora do ambiente isolado (o `rodar_tudo.sh` já usa `python3`).

---

## 3. Contagem no dataset

| Conferência | EN | PT | Total |
|---|---|---|---|
| brasnam | 33 | 107 | 140 |
| eniac | 287 | 265 | 552 |
| sbbd | 155 | 183 | 338 |
| sbrc | 66 | 309 | 375 |
| semish | 58 | 132 | 190 |
| wics | 17 | 90 | 107 |
| Total | 616 | 1086 | 1702 |

| Ano | EN | PT | Total |
|---|---|---|---|
| 2020 | 68 | 159 | 227 |
| 2021 | 78 | 166 | 244 |
| 2022 | 63 | 148 | 211 |
| 2023 | 87 | 172 | 259 |
| 2024 | 128 | 188 | 316 |
| 2025 | 192 | 253 | 445 |
| Total | 616 | 1086 | 1702 |

**Unidades de texto analisadas** (`src/scripts_auxiliares/textos.py`):
- **corpo** — conteúdo recursivo das seções, **sem** Agradecimentos/Acknowledgments/Apêndices/Anexos (377 seções, 375 delas a última do artigo, mediana de 207 caracteres) e **sem** seções de referências que sejam de fato listas bibliográficas (só 3). *Atenção:* das 146 seções com título "Referências/References", **143 são texto do próprio artigo** (introdução, trabalhos relacionados…) rotulado errado pelo parser, em geral no meio do documento; por isso **não** se exclui por título, só por conteúdo (`bib_score` ≥ 15 sinais bibliográficos por 1.000 caracteres)..
- **resumo** — abstract no idioma do artigo (`abstract[lang]`); 30 artigos sem resumo utilizável nesse idioma ficam sem valores (31 no CSV de métricas). **O resumo é calculado, mas as análises e figuras principais usam o corpo.**

**Observações sobre os dados.**
- **Conferências com fórmulas e código (SBBD):** artigos de bancos de dados trazem muitas fórmulas, diagramas, URLs e código SQL, o que eleva a % de símbolos e o bits/caractere (menor previsibilidade) e torna os artigos mais curtos em palavras (SBBD 2020: mediana de 2.340 palavras em EN e 1.890 em PT, contra ~3.500–3.700 nos demais artigos do período). Isso foi verificado manualmente nos artigos (2026-10-16) e **é característica da conferência, não ruído de OCR**: nenhum artigo é excluído ou analisado à parte, e as diferenças de nível entre conferências devem ser atribuídas à natureza das conferências e de seus artigos (o ajuste por conferência na seção 10 trata dessa heterogeneidade).
- **Resumos:** 31 artigos sem resumo no idioma do artigo (2 sem resumo algum).
- Células pequenas: WICS em inglês (2–5 artigos por ano; nenhum em 2021), BraSNAM e SBRC em inglês em alguns anos (1–9).

---

## 4. Scripts e parâmetros (`src/`)

**Pipeline principal (ordem do `rodar_tudo.sh`):**

| # | Script | Função | Entradas → saídas | Parâmetros principais |
|---|---|---|---|---|
| 0 | `filtra_datasets.py` | monta o dataset (seção 3) | `dataset/converted_json` → `dataset/english/`, `dataset/portuguese/`, `resultados/canonico/{lista_canonica,excluidos}.csv` | `EXCLUIR` (5 arquivos); langdetect seed 0 no corpo; duplicata: TF-IDF `max_features=50000, sublinear_tf`, cosseno > 0,8; `curto` < 10.000 caracteres |
| 1 | `calcula_metricas.py` (+ `metricas.py`, `textos.py`, `marcadores.py`) | **script oficial das métricas** (29 colunas) para `corpo` e `resumo` | datasets → `resultados/metricas/metricas_{corpo,resumo}.csv`, `config.json` | `--unidades corpo resumo`, `--limit N`, `--sem-lm`, `--refaz`; spaCy `*_core_*_sm` (sem NER/lematizador); LM: Llama-3.2-3B (fp16) e GPT-2 por idioma, janelas de **1.023 tokens** (+BOS) sem sobreposição, mínimo 32 tokens; `hdd draws=42`; textos com < 20 palavras ficam vazios |
| 2 | `calcula_pangram.py` (+ `pangram.py`) | Pangram-Llama no **corpo** | datasets → `resultados/pangram/pangram_{janelas,secoes,artigos}.csv`, `config.json` | janela **512 tokens** sem sobreposição dentro de cada seção, mínimo 32, `BATCH=8`; agregação = média das probabilidades ponderada por tokens; classe = argmax; `--limit`, `--refaz` |
| 3 | `calcula_pangram_resumo.py` | Pangram-Llama no resumo | datasets → `resultados/pangram/pangram_resumo.csv` | mesma janela (512; o resumo cabe em 1 janela em 97,5% dos casos) |
| 4 | `secoes.py` | mapeia títulos de seção → 5 categorias (+ "Outras") e agrega o Pangram por artigo × categoria | `pangram_janelas.csv`, `pangram_resumo.csv` → `pangram_artigo_categoria.csv`, `resultados/analise/mapeamento_secoes.csv` | regras por regex PT/EN (no próprio arquivo); `Preamble` = Introdução; ~84% dos tokens classificados |
| 5 | `correlacoes.py`, `correlacoes_por_ano.py`, `tabela_categorias.py` | Spearman métrica × Pangram (pooled, por ano, pré/pós, "intra-ano") e tabela por categoria | métricas + Pangram → `resultados/analise/correlacao_*.csv`, `tabela_por_categoria.{csv,md}` | escore do Pangram `E = (P1 + 2·P2 + 3·P3)/300`; Holm sobre as colunas analisadas por idioma |
| 6 | `testes_pre_pos.py` | testes pré × pós | métricas → `testes_pre_pos.{csv,md}` | Mann-Whitney bilateral; bootstrap 2.000 reamostragens (seed 0) para o IC do d; família = 13 × 2 = 26; Holm e Benjamini-Hochberg; 2 sensibilidades: sem 2023 e ajuste por conferência (seção 10) |

---

## 5. Métricas avaliadas

Foram calculadas **29 colunas** em `metricas_corpo.csv` (13 mantidas, 14 descartadas, 2 de controle) e testadas, além disso, variantes exploratórias (7.3). Critério de descarte (pedido do usuário): **redundância entre métricas ou fragilidade conceitual, independente da correlação com o Pangram** (para que se possa mostrar aspectos linguísticos independentes da detecção).

### 5.1 Métricas MANTIDAS nas análises (13) + controle de tamanho
| Categoria | Coluna | Definição / parâmetros | Leitura |
|---|---|---|---|
| Complexidade lexical | `mtld` | Diversidade lexical MTLD (McCarthy & Jarvis, 2010): comprimento médio de sequências de palavras até o TTR cair a 0,72 (`lexicalrichness`, texto do corpo; palavras = tokens da biblioteca) | maior = vocabulário mais diverso |
| Complexidade lexical | `hdd` | HD-D (McCarthy & Jarvis, 2007): número esperado de tipos distintos em amostras de 42 palavras (`lexicalrichness`, `draws=42`) | maior = vocabulário mais diverso |
| Complexidade lexical | `lexical_density` | Fração das palavras alfabéticas que são de conteúdo (NOUN, PROPN, VERB, ADJ, ADV) segundo o spaCy (`pt_core_news_sm`/`en_core_web_sm`) | maior = texto mais denso em conteúdo |
| Previsibilidade | `llama_bpc` | Bits por caractere sob o Llama-3.2-3B (o mesmo modelo nos dois idiomas): soma da surpresa (−log₂ p) de todos os tokens do corpo ÷ nº de caracteres; texto inteiro em janelas sem sobreposição de 1.023 tokens (+BOS), janelas < 32 tokens descartadas | menor = texto mais previsível |
| Previsibilidade | `llama_surp_sd` | Desvio-padrão da surpresa por token (nats) sob o Llama-3.2-3B, sobre todos os tokens do corpo | maior = surpresa mais variável |
| Estilo | `burstiness` | σ/µ (desvio-padrão populacional ÷ média) do nº de palavras alfabéticas por frase; frases segmentadas pelo spaCy; só frases com > 3 palavras | menor = tamanho de frase mais regular |
| Estilo | `bigram_rep_per1k` | Bigramas repetidos por 1.000 bigramas: 1000 × média, em janelas sem sobreposição de 1.000 palavras, de (1 − bigramas únicos/total); NaN se o corpo tem < 1.000 palavras | maior = mais repetição |
| Estilo | `ai_marker_per1k` | Ocorrências de marcadores lexicais associados a LLMs por 1.000 palavras (palavra/expressão inteira, sem diferenciar maiúsculas); listas em `src/marcadores.py` | maior = mais marcadores |
| Estilo | `hedge_per1k` | Ocorrências de atenuadores epistêmicos por 1.000 palavras (mesmo método); listas em `src/marcadores.py` | maior = mais atenuação |
| Legibilidade | `fre` | Flesch Reading Ease = base − 1,015·(palavras/frase) − 84,6·(sílabas/palavra); base 206,835 (EN) ou 248,835 (PT, Martins et al., 1996); sílabas por `pyphen` (pt_BR / en_US); frases do spaCy | maior = mais fácil de ler |
| Sintaxe | `mdd` | Distância média de dependência (Liu, 2008): média, por frase (≥ 4 tokens), da distância |posição do token − posição do núcleo| (exclui a raiz), e média entre as frases | maior = dependências mais longas |
| Sintaxe | `connector_per1k` | Conectores por 1.000 palavras: tokens com classe CCONJ/SCONJ ou relação de dependência `cc`/`mark` (spaCy) | maior = mais conjunções |
| Sintaxe | `sent_len_mean` | Média de palavras alfabéticas por frase (frases do spaCy com > 3 palavras) | maior = frases mais longas |

Controle de tamanho (não é métrica linguística): `n_words` (palavras alfabéticas do corpo) e `n_sentences` (frases do spaCy com ≥ 1 palavra). Todas as métricas só devem ser comparadas **dentro do mesmo idioma** quando dependem de modelo/lista de cada língua (`lexical_density`, `connector_per1k`, `mdd`, `fre`, `ai_marker_per1k`, `hedge_per1k`, `llama_*`).

### 5.2 Métricas DESCARTADAS das análises (14; continuam no CSV e no script de geração)
| Coluna | O que mede | Motivo do descarte (independe da correlação com o Pangram) |
|---|---|---|
| `yules_k` | Índice K de Yule: 10⁴·(M2 − N)/N², tokens alfabéticos em minúsculas | redundante com HD-D (ρ −0,97 EN / −0,92 PT), sinal invertido e tokenização original não reproduzida |
| `mattr100` | TTR médio em janelas móveis de 100 palavras (MATTR) | redundante com MTLD (ρ 0,94 EN / 0,93 PT); MTLD é a medida tradicional |
| `llama_ppl` | Perplexidade por token sob o Llama-3.2-3B | é o bpc em outra escala e depende do tokenizador (EN 11,2 vs PT 7,0, o oposto do bpc); bpc = perplexidade normalizada por caractere |
| `gpt2_bpc` | Bits por caractere sob GPT-2 (gpt2 em EN; pierreguillou/gpt2-small-portuguese em PT) | modelo de controle descartado como análise; GPT-2 e Llama concordam (ρ 0,80 EN / 0,76 PT); mencionar como limitação/trabalho relacionado |
| `gpt2_ppl` | Perplexidade por token sob o GPT-2 do idioma | dois GPT-2 diferentes por idioma (não comparáveis entre idiomas); mais fraca da categoria |
| `gpt2_surp_sd` | Desvio da surpresa por token sob o GPT-2 do idioma | mais fraca e menos relacionada às demais (ρ 0,48-0,55 com o Llama) |
| `bigram_rep` | 1 − (bigramas únicos/total) sobre o corpo inteiro (fração, depende do comprimento) | fração bruta de bigramas repetidos; depende do comprimento (ρ com n_words 0,49 EN / 0,60 PT); substituída por bigram_rep_per1k (ρ 0,88/0,86 entre as duas) |
| `ai_marker_count` | Contagem bruta de marcadores de IA | contagem bruta de marcadores; redundante com ai_marker_per1k (ρ 0,95/0,96) e dependente do comprimento |
| `hedge_count` | Contagem bruta de atenuadores | contagem bruta de atenuadores; redundante com hedge_per1k (ρ 0,92/0,94) e dependente do comprimento |
| `fk_grade` | Flesch-Kincaid grade level (0,39·palavras/frase + 11,8·sílabas/palavra − 15,59); só EN (NaN em PT) | existe só para EN (sem adaptação validada para PT); mesmas entradas do fre (tamanho de frase e de palavra), logo redundante com ele; fre cobre os dois idiomas |
| `depth_mean` | Profundidade máxima da árvore sintática por frase, média entre frases | profundidade da árvore muito ligada ao tamanho da frase (ρ 0,80 EN / 0,86 PT com sent_len_mean); pouco acrescenta além de sent_len_mean e mdd |
| `depth_p90` | Idem, percentil 90 entre frases | redundante com depth_mean (ρ 0,84/0,79) e com sent_len_mean (0,67/0,71) |
| `sent_len_sd` | Desvio-padrão do nº de palavras por frase | burstiness sem normalizar pela média (ρ 0,86 EN / 0,68 PT); burstiness já está em Estilo |
| `ttr` | Type-Token Ratio (tipos/tokens) do corpo | depende do comprimento do texto (ρ com n_words −0,68 EN / −0,74 PT); MTLD e HD-D medem a mesma ideia sem esse viés |
---

## 6. Listas de marcadores e atenuadores (`src/scripts_auxiliares/marcadores.py`)

- **Origem:** as listas originais dos coautores (`referencias/ai_markers_and_hedges_original.py`; EN 44 marcadores e 14 atenuadores, PT 50 e 20; **reproduzem 100%** das contagens do CSV antigo). As atuais são um subconjunto/adaptação, definido por critério de **respaldo na literatura**.
- **Marcadores em inglês (18 famílias, 33 formas):** os itens da lista original presentes (forma exata ou mesma raiz) na lista de excesso de vocabulário de **Kobak et al. (2025)**: delve, leverage, pivotal, crucial, underscore, foster, realm, landscape, commendable, noteworthy, additionally, consequently, comprehensive, thorough, showcase, multifaceted, nuanced, groundbreaking. Respaldo adicional: Juzek & Ward (2025; delve, underscore, showcase, realm, groundbreaking) e Geng & Trotta (2025, citando Liang et al.; commendable, pivotal, comprehensive, crucial). **Saíram** (sem respaldo encontrado nas fontes lidas): vital, paramount, synergy, furthermore, moreover, robust, holistic e as frases/hifenizadas (it is worth noting…, shed light, in the context of, cutting-edge, state-of-the-art).
- **Marcadores em português (62 formas):** correspondentes diretas, com flexões de gênero/número/pessoa, das famílias em inglês que já existiam na lista original em português (aprofundar, alavancar, abrangente, multifacetado, notável) **mais** os exemplos que aparecem no texto do esboço (fundamental, crucial, fomentar, âmbito, panorama, sutil, ressaltar, inovador, sinergia, "de vanguarda"). Sem respaldo encontrado: sinergia e "de vanguarda" (entraram por serem do esboço). Itens só da lista original PT **fora** da atual: robusto, holístico, sinérgico, paradigmático, "de ponta", primordial, "vale ressaltar/destacar", "no contexto de", transformador e pioneiro (estes dois têm respaldo em Kobak: *transformative*, *pioneering*).
- **Atenuadores:** EN = lista original completa (14); PT = correspondentes na lista original (poderia/poderiam, possivelmente, talvez, parece/parecem, tende/tendem a, geralmente, tipicamente, em geral, sugere/sugerem). "parece que" foi removido (⊂ "parece", contaria duas vezes).
- **Pendente** (documentado em `marcadores.PENDENTES_*`): equivalentes em PT para 6 famílias em inglês (commendable→louvável, additionally→adicionalmente, consequently→consequentemente, thorough→minucioso, showcase, groundbreaking); "pode/podem" (equivalente de *may*, o atenuador mais frequente em EN); flexões em EN dos atenuadores (appears/seems/tends to); **divergência entre o esboço (30 itens) e o arquivo (44/50 entradas)** — confirmar qual lista é a do artigo.
- **Efeito da troca de listas:** `ai_marker_per1k` EN passou a correlacionar ρ 0,70 com o Pangram (0,63 com a lista reconstruída anterior); PT 0,44; artigos PT sem nenhum marcador: 21%.

## 7. Pangram-Llama
- **Modelo:** EditLens oficial da Pangram sobre Llama-3.2-3B (`pangram/editlens_Llama-3.2-3B`, adapter QLoRA 4-bit sobre `meta-llama/Llama-3.2-3B`, cabeça LayerNorm→Linear de 4 classes), carregado conforme o `scripts/inference.py` do repositório EditLens. Saída: probabilidades (%) das 4 classes: 0 humano, 1 pouco editado por IA, 2 muito editado por IA, 3 gerado por IA (nomes conforme o artigo do EditLens; **confirmar**).
- **Método:** corpo em janelas de 512 tokens sem sobreposição **dentro de cada seção**; média das probabilidades ponderada por tokens por seção e por artigo; classe = argmax; `escore_E = (P1 + 2·P2 + 3·P3)/300`; `frac_janelas_classe23`. 23.679 janelas, 9.845 seções.
- **Por que janelas de 512:** a pre-análise pontuou seções inteiras (sem chunking) e o texto completo de forma consistente com janelas de 256–512 tokens; o modelo é calibrado em trechos curtos — textos longos deslocam as probabilidades e embaralham o ranking (tabela 7.3).
- **Concordância com o Pangram-Llama da pre-análise (artigo, texto completo):** Spearman P0 0,93, P1 0,78, P2 0,83, P3 0,91; médias quase iguais.
- **Nível basal alto, sobretudo em PT:** em 2020–22, 16–18% das janelas em português já caem nas classes 2–3 (EN: 1–2%); só a variação ao longo do tempo é informativa, não o valor absoluto. Textos curtos (resumo, conclusão) pontuam mais alto (viés conhecido do detector).

Médias do corpo por ano (todas as conferências):

| Ano | EN: P0 (humano) % | EN: P3 (IA) % | PT: P0 (humano) % | PT: P3 (IA) % |
|---|---|---|---|---|
| 2020.0 | 64.2 | 5.2 | 53.6 | 13.9 |
| 2021.0 | 58.5 | 7.1 | 52.0 | 14.7 |
| 2022.0 | 60.2 | 6.3 | 51.3 | 15.1 |
| 2023.0 | 41.4 | 14.0 | 48.1 | 17.1 |
| 2024.0 | 27.6 | 21.5 | 40.2 | 22.3 |
| 2025.0 | 15.5 | 35.9 | 27.3 | 33.5 |

## 8. Estatística (`testes_pre_pos.py`)
- **Teste:** Mann-Whitney U bilateral por métrica e idioma, **2020–22 (pré) × 2023–25 (pós)** (o esboço prevê Mann-Whitney, α = 0,05).
- **Tamanhos de efeito:** d de Cohen (desvio combinado) com IC 95% por bootstrap (2.000 reamostragens, seed 0), rank-biserial (delta de Cliff), Δ das medianas.
- **Correção para múltiplos testes:** família = 13 métricas × 2 idiomas = **26 testes**; **Holm** (controla qualquer falso positivo — critério principal) e **Benjamini-Hochberg** (FDR; reportado no CSV). Sem correção, P(≥ 1 falso positivo) = 74%. A correção **não alterou nenhuma conclusão** (10/13 significativas por idioma com p bruto, Holm ou BH).
- **Sensibilidades** (cada uma com Holm sobre os 26 testes): (a) sem 2023 (ano de transição; compara 2020–22 com 2024–25); (b) ajuste por conferência (OLS da métrica padronizada ~ pós + conferência, erros-padrão robustos HC3; o coeficiente de "pós" é a diferença antes × depois *dentro da mesma conferência*, em desvios-padrão). **"Robusto" = significativo (Holm < 0,05) e com o mesmo sinal na análise principal e nas duas sensibilidades.** Nenhum artigo é excluído.
- Linhas de controle (`n_words`, `n_sentences`) e a referência (`escore_E`) são reportadas fora da família.

## 9. Resultados de referência
Mudança entre 2020–22 e 2023–25 (d de Cohen, positivo = maior no pós; p ajustado por Holm; "robusto" conforme a seção 10):

| Categoria | Métrica | EN d | EN p Holm | EN robusto | PT d | PT p Holm | PT robusto |
|---|---|---|---|---|---|---|---|
| Complexidade lexical | `mtld` | +0,60 | <0,001 | sim | +0,51 | <0,001 | sim |
| Complexidade lexical | `hdd` | +0,69 | <0,001 | sim | +0,30 | <0,001 | sim |
| Complexidade lexical | `lexical_density` | +0,93 | <0,001 | sim | +0,57 | <0,001 | sim |
| Previsibilidade | `llama_bpc` | -0,93 | <0,001 | sim | -0,31 | <0,001 | sim |
| Previsibilidade | `llama_surp_sd` | -0,57 | <0,001 | sim | -0,23 | 0,012 | sim |
| Estilo | `burstiness` | -0,61 | <0,001 | sim | -0,32 | <0,001 | sim |
| Estilo | `bigram_rep_per1k` | -0,41 | <0,001 | sim | -0,39 | <0,001 | sim |
| Estilo | `ai_marker_per1k` | +1,12 | <0,001 | sim | +0,63 | <0,001 | sim |
| Estilo | `hedge_per1k` | +0,06 | 0,945 | não | +0,08 | 0,471 | não |
| Legibilidade | `fre` | -0,93 | <0,001 | sim | -0,61 | <0,001 | sim |
| Sintaxe | `mdd` | +0,23 | 0,005 | não | -0,03 | 1,000 | não |
| Sintaxe | `connector_per1k` | +0,06 | 0,945 | não | +0,21 | 0,012 | sim |
| Sintaxe | `sent_len_mean` | -0,02 | 1,000 | não | +0,04 | 0,945 | não |

- **Robustas: 9 de 13 em EN e 10 de 13 em PT.** Única significativa mas não robusta: `mdd` em EN (perde significância ao ajustar por conferência, Holm 0,074: a mudança vem de quais conferências cresceram, não de mudança dentro delas). Sem mudança em nenhum idioma: atenuadores e palavras por frase.
- O escore do próprio Pangram muda mais que qualquer métrica linguística: d = +1,83 (EN) e +1,24 (PT).
- Correlação métrica × Pangram (Spearman, pooled): melhores — EN: `fre` −0,68, `ai_marker_per1k` 0,70, `llama_bpc` −0,63, `lexical_density` 0,61, `mtld` 0,59; PT: `fre` −0,57, `mtld` 0,49, `llama_bpc` −0,41, `lexical_density` 0,40. Controlando o ano ("intra-ano") o sinal cai 25–40%, e **em 2020–22 `fre`, `mtld` e bigramas já correlacionam com o Pangram** (viés de estilo do detector: descrevem mudança de estilo mas não distinguem IA de texto formal); `llama_bpc`, `burstiness` e `lexical_density` só correlacionam depois de 2023 em inglês.
- Por seção (Pangram, média 2020–22 → 2023–25): Conclusão EN 0,24 → 0,70, PT 0,44 → 0,65; Resumo EN 0,27 → 0,66, PT 0,43 → 0,65; Metodologia e Resultados sobem menos (EN 0,19–0,20 → 0,45–0,48).
- Figuras e tabelas em `resultados_referencia/analise/` (forest plot, métricas por categoria EN|PT, por conferência com e sem DP, escore por seção/ano/conferência, classes 0 e 3).
