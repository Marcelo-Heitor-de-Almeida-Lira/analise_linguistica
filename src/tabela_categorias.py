"""Tabela de correlação métrica × Pangram agrupada por categoria linguística, ordenada por |ρ(E)| em inglês dentro de cada categoria.
Lê resultados/analise/correlacao_metricas_pangram.csv e correlacao_por_ano.csv; grava resultados/analise/tabela_por_categoria.{csv,md}.
Uso: python3 -I src/tabela_categorias.py"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts_auxiliares.config_metricas import CATEGORIAS as CAT  # noqa: E402
a = pd.read_csv(ROOT / "resultados/analise/correlacao_metricas_pangram.csv")
b = pd.read_csv(ROOT / "resultados/analise/correlacao_por_ano.csv")
sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
rows = []
for cat, ms in CAT.items():
    sub = []
    for m in ms:
        r = dict(categoria=cat, metrica=m)
        for lg in ("en", "pt"):
            x = a[(a.idioma == lg) & (a.metrica == m)]
            y = b[(b.idioma == lg) & (b.metrica == m)]
            r[f"rho_E_{lg}"] = x.rho_E.iloc[0] if len(x) else np.nan
            r[f"sig_{lg}"] = sig(x.p_holm.iloc[0]) if len(x) else ""
            r[f"rho_P3_{lg}"] = x.rho_P3.iloc[0] if len(x) else np.nan
            r[f"intra_{lg}"] = y.intra_ano.iloc[0] if len(y) else np.nan
        sub.append(r)
    rows += sorted(sub, key=lambda r: -abs(r["rho_E_en"]))
t = pd.DataFrame(rows)
t.to_csv(ROOT / "resultados/analise/tabela_por_categoria.csv", index=False)
f = lambda v: "n/d" if pd.isna(v) else f"{v:+.2f}".replace("+", "").replace(".", ",")
md = ["| Categoria | Métrica | ρ(E) EN | sig | ρ(P3) EN | intra-ano EN | ρ(E) PT | sig | ρ(P3) PT | intra-ano PT |", "|---|---|---|---|---|---|---|---|---|---|"]
ult = None
for r in rows:
    md.append(f"| {r['categoria'] if r['categoria'] != ult else ''} | {r['metrica']} | {f(r['rho_E_en'])} | {r['sig_en']} | {f(r['rho_P3_en'])} | {f(r['intra_en'])} | {f(r['rho_E_pt'])} | {r['sig_pt'] or 'n/d'} | {f(r['rho_P3_pt'])} | {f(r['intra_pt'])} |")
    ult = r["categoria"]
(ROOT / "resultados/analise/tabela_por_categoria.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
