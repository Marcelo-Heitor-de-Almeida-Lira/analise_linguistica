"""Correlação de Spearman (métrica × escore_E do Pangram) POR ANO e por idioma, e correlação 'intra-ano' (remove a deriva temporal comum):
os valores de cada variável são convertidos em postos dentro do ano e depois correlacionados (equivale a controlar o ano).
Saída: resultados/analise/correlacao_por_ano.csv.  Uso: python3 -I src/correlacoes_por_ano.py"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resultados" / "analise"
OUT.mkdir(parents=True, exist_ok=True)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts_auxiliares.config_metricas import METRICAS  # noqa: E402
m = pd.read_csv(ROOT / "resultados/metricas/metricas_corpo.csv")
p = pd.read_csv(ROOT / "resultados/pangram/pangram_artigos.csv")[["source_file", "escore_E"]]
d = m.merge(p, on="source_file")
skip = {"source_file", "event", "year", "article_id", "language", "escore_E"}
cols = [c for c in METRICAS if c in m.columns]  # seleção de config_metricas.py (exclui yules_k, mattr100)
rows = []
for lang, g in d.groupby("language"):
    for c in cols:
        x = g[["year", c, "escore_E"]].dropna()
        if len(x) < 50:
            continue
        r = dict(idioma=lang, metrica=c)
        for y, gy in x.groupby("year"):
            r[str(y)] = spearmanr(gy[c], gy.escore_E)[0]
        rk = x.assign(a=x.groupby("year")[c].rank(pct=True), b=x.groupby("year").escore_E.rank(pct=True))
        r["intra_ano"], r["p_intra"] = spearmanr(rk.a, rk.b)
        pre, pos = x[x.year <= 2022], x[x.year >= 2023]
        r["pre_2020_22"] = spearmanr(pre[c], pre.escore_E)[0]
        r["pos_2023_25"] = spearmanr(pos[c], pos.escore_E)[0]
        r["pooled"] = spearmanr(x[c], x.escore_E)[0]
        rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(OUT / "correlacao_por_ano.csv", index=False)
pd.set_option("display.width", 220)
for lang in ("en", "pt"):
    g = t[t.idioma == lang].drop(columns=["idioma", "p_intra"]).set_index("metrica")
    g = g[["pooled", "intra_ano", "pre_2020_22", "pos_2023_25", "2020", "2021", "2022", "2023", "2024", "2025"]]
    print(f"\n=== {lang.upper()} — ordenado por |intra_ano|  (n por ano: {d[d.language == lang].groupby('year').size().to_dict()})")
    print(g.reindex(g.intra_ano.abs().sort_values(ascending=False).index).round(2).to_string())
