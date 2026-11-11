#!/usr/bin/env bash
# Reproduz o pipeline completo a partir da raiz do pacote (src/). Ver README.md para entradas, saídas e parâmetros.
# Tempo aproximado em RTX 4070 SUPER (sem contabilizar o Web-scraping): métricas ~40 min, Pangram (corpo) ~40 min, Pangram (resumo) ~2 min, demais etapas poucos minutos.
# Use "python3" sem -I (os scripts importam pacotes instalados no ambiente). Pule as etapas que não quiser repetir.
set -euo pipefail
cd "$(dirname "$0")"
PY=python3

echo "== 0. web-scraping (HTMLs -> (JSONs, PDFs))";                    $PY web_scraping/scraper.py           # CPU com multithreading;
echo "== 1. conversão de PDFs -> JSONs";                               $PY web_scraping/pdf_to_json_grobid.py
echo "== 2. dataset: JSONs estruturados -> dataset_pt/, dataset_en/";  $PY src/filtra_datasets.py            # CPU; ~1 min
echo "== 3. métricas linguísticas (corpo e resumo)";                   $PY src/calcula_metricas.py --refaz   # GPU (Llama-3.2-3B + GPT-2); ~40 min
echo "== 4. Pangram-Llama no corpo (janelas de 512 tokens)";           $PY src/calcula_pangram.py --refaz    # GPU + HF_TOKEN2; ~40 min
echo "== 5. Pangram-Llama no resumo";                                  $PY src/calcula_pangram_resumo.py     # GPU + HF_TOKEN2; ~2 min
echo "== 6. categorias de seção (agrega o Pangram por artigo x seção)"; $PY src/secoes.py
echo "== 7. correlações métrica x Pangram";                            $PY src/correlacoes.py; $PY src/correlacoes_por_ano.py; $PY src/tabela_categorias.py
echo "== 8. testes pré x pós (Mann-Whitney, Holm, BH, sensibilidades)"; $PY src/testes_pre_pos.py
echo "== 9. gráficos"
for g in grafico_conferencias grafico_secoes grafico_labels grafico_metricas grafico_metricas_conferencia grafico_metricas_conferencia_dp grafico_forest; do $PY src/scripts_graficos/$g.py; done
echo "Pronto. Saídas em resultados/ (compare com resultados_referencia/)."
