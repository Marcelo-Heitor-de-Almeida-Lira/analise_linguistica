"""SCRIPT OFICIAL do Pangram-Llama sobre o CORPO dos artigos (v1.0 — decisão de 2026-10-15, research.md §8).

Para cada artigo de dataset/portuguese/ e dataset/english/: cada seção do corpo (textos.py: sem agradecimentos/bibliografia real) é dividida em janelas SEM
sobreposição de 512 tokens (janelas < 32 tokens são descartadas); cada janela é pontuada pelo Pangram-Llama (4 probabilidades, em %).
Saídas em resultados/pangram/:
  pangram_janelas.csv : uma linha por janela (article, secao_idx, secao_titulo, janela_idx, n_tokens, p0..p3)
  pangram_secoes.csv  : por seção — média das probabilidades (ponderada por tokens), classe = argmax da média, n_janelas, n_tokens
  pangram_artigos.csv : por artigo — idem sobre todas as janelas do corpo; + escore E=(p1+2·p2+3·p3)/300 e fração de janelas cuja classe é 2 ou 3
Classes (EditLens): 0 = humano, 1 = pouco editado por IA, 2 = muito editado por IA, 3 = gerado por IA (nomes conforme o artigo do EditLens; confirmar).
Retomável (pula artigos já presentes em pangram_janelas.csv). Uso: python3 src/calcula_pangram.py [--limit N] [--refaz]  (GPU, sem -I)"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scripts_auxiliares.pangram as pangram  # noqa: E402
import scripts_auxiliares.textos as textos  # noqa: E402

JANELA, MIN_TOK, BATCH = 512, 32, 8
NOMES = {0: "humano", 1: "pouco editado por IA", 2: "muito editado por IA", 3: "gerado por IA"}
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resultados" / "pangram"
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int)
ap.add_argument("--refaz", action="store_true")
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
jan_path = OUT / "pangram_janelas.csv"
if a.refaz and jan_path.exists():
    jan_path.unlink()
tok, _ = pangram.load()
feitos = set(pd.read_csv(jan_path).source_file) if jan_path.exists() else set()
files = [(l, p) for l in ("portuguese", "english") for p in sorted((ROOT / f"dataset/{l}").glob("*/*/*.json"))]
if a.limit:
    files = files[: a.limit]
COLS = ["source_file", "secao_idx", "secao_titulo", "janela_idx", "n_tokens", "p0", "p1", "p2", "p3"]
buf = []


def flush():
    global buf
    if buf:
        pd.DataFrame(buf, columns=COLS).to_csv(jan_path, mode="a", header=not jan_path.exists(), index=False)
        buf = []


for i, (lang, p) in enumerate(files):
    if p.name in feitos:
        continue
    j = json.load(open(p, encoding="utf-8"))
    trechos = []  # (secao_idx, titulo, janela_idx, texto, n_tokens)
    for si, sec in enumerate(s for s in j["sections"] if not textos.excluir(s)):
        txt = textos._conteudo(sec)
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        for wi, k in enumerate(range(0, len(ids), JANELA)):
            c = ids[k:k + JANELA]
            if len(c) >= MIN_TOK:
                trechos.append((si, sec.get("sectionTitle") or "", wi, tok.decode(c), len(c)))
    if trechos:
        pr = pangram.probs([t[3] for t in trechos], batch=BATCH, max_length=JANELA + 4)
        for t, q in zip(trechos, pr):
            buf.append([p.name, t[0], t[1], t[2], t[4], *np.round(q, 4)])
    if (i + 1) % 25 == 0:
        flush(); print(i + 1, "/", len(files), flush=True)
flush()

# ---- agregação ----
w = pd.read_csv(jan_path)
P = ["p0", "p1", "p2", "p3"]


def agrega(g):
    pesos = g.n_tokens.values
    m = np.average(g[P].values, axis=0, weights=pesos)
    return pd.Series(dict(n_janelas=len(g), n_tokens=int(pesos.sum()), **{f"P{k}": round(m[k], 3) for k in range(4)},
                          classe=int(np.argmax(m)), classe_nome=NOMES[int(np.argmax(m))]))


sec = w.groupby(["source_file", "secao_idx", "secao_titulo"]).apply(agrega, include_groups=False).reset_index()
art = w.groupby("source_file").apply(agrega, include_groups=False).reset_index()
cls = w[P].values.argmax(1)
art["frac_janelas_classe23"] = art.source_file.map(pd.Series(cls >= 2).groupby(w.source_file).mean())
art["escore_E"] = (art.P1 + 2 * art.P2 + 3 * art.P3) / 300
meta = pd.DataFrame([dict(source_file=p.name, event=p.stem.split("_")[0], year=int(p.stem.split("_")[1]), article_id=int(p.stem.split("_")[2]), language=l) for l, p in files])
art = meta.merge(art, on="source_file")
sec = meta.merge(sec, on="source_file")
sec.to_csv(OUT / "pangram_secoes.csv", index=False)
art.to_csv(OUT / "pangram_artigos.csv", index=False)
(OUT / "config.json").write_text(json.dumps({"modelo": pangram.ADAPTER, "base": pangram.BASE, "janela_tokens": JANELA, "min_tokens": MIN_TOK,
                                             "agregacao": "média das probabilidades ponderada por tokens", "classes": NOMES}, indent=1, ensure_ascii=False))
print("ok", len(w), "janelas |", len(sec), "seções |", len(art), "artigos")
