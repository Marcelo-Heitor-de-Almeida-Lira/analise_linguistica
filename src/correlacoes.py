"""Correlação de Spearman de cada métrica linguística (corpo) com o Pangram-Llama (corpo), por idioma.
Pangram por artigo: escore_E (0-1, esperado das 4 classes), P3 (% IA), P0 (% humano). Saída: resultados/analise/correlacao_metricas_pangram.csv
Uso: python3 -I src/correlacoes.py"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LANG_DIR = {"portuguese": "pt", "english": "en"}

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resultados" / "analise"
OUT.mkdir(parents=True, exist_ok=True)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts_auxiliares.config_metricas import METRICAS  # noqa: E402
m = pd.read_csv(ROOT / "resultados/metricas/metricas_corpo.csv")
p = pd.read_csv(ROOT / "resultados/pangram/pangram_artigos.csv")[["source_file", "escore_E", "P0", "P3"]]
d = m.merge(p, on="source_file")
skip = {"source_file", "event", "year", "article_id", "language", "escore_E", "P0", "P3"}
cols = [c for c in METRICAS if c in m.columns]  # seleção de config_metricas.py (exclui yules_k, mattr100)
rows = []
for language, g in d.groupby("language"):
    lang = LANG_DIR[language]
    for c in cols:
        x = g[[c, "escore_E", "P0", "P3"]].dropna()
        print(x)
        if len(x) < 30:
            continue
        r = {k: spearmanr(x[c], x[k]) for k in ("escore_E", "P3", "P0")}
        rows.append(dict(idioma=lang, metrica=c, n=len(x), rho_E=r["escore_E"][0], p_E=r["escore_E"][1], rho_P3=r["P3"][0], rho_P0=r["P0"][0]))
t = pd.DataFrame(rows)
# Holm por idioma (sobre p_E)
t["p_holm"] = np.nan
for lang, g in t.groupby("idioma"):
    o = g.p_E.sort_values().index; k = len(g); run = 0
    for rank, i in enumerate(o):
        run = max(run, (k - rank) * t.loc[i, "p_E"]); t.loc[i, "p_holm"] = min(1, run)
t.to_csv(OUT / "correlacao_metricas_pangram.csv", index=False)
w = t.pivot(index="metrica", columns="idioma", values=["rho_E", "rho_P3"]).round(2)
sig = t.assign(s=np.where(t.p_holm < 0.001, "***", np.where(t.p_holm < 0.01, "**", np.where(t.p_holm < 0.05, "*", "ns")))).pivot(index="metrica", columns="idioma", values="s")
out = pd.DataFrame({"ρ(E) EN": w[("rho_E", "en")], "sig EN": sig["en"], "ρ(P3) EN": w[("rho_P3", "en")],
                    "ρ(E) PT": w[("rho_E", "pt")], "sig PT": sig["pt"], "ρ(P3) PT": w[("rho_P3", "pt")]})
out = out.reindex(out["ρ(E) EN"].abs().sort_values(ascending=False).index)
print(out.to_string())
print("\nn EN =", int((d.language == "english").sum()), "| n PT =", int((d.language == "portuguese").sum()))
