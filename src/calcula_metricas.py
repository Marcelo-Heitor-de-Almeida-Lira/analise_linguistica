"""SCRIPT OFICIAL de geração das métricas (v1.0 — ver metricas.py e research.md §6c).

Lê os JSONs estruturados de dataset/english/ e dataset/portuguese/ e grava, por unidade de texto,
resultados/metricas/metricas_<unidade>.csv  (unidade = corpo | resumo), com colunas
source_file, event, year, article_id, language + as métricas de metricas.compute().
Retomável: reaproveita linhas já calculadas se o CSV existir (use --refaz para recomeçar).

Uso:  python3 src/calcula_metricas.py [--unidades corpo resumo] [--limit N] [--sem-lm] [--refaz]
(sem -I: transformers/spacy/langdetect estão no site-packages do usuário; precisa de GPU para os LMs)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scripts_auxiliares.metricas as metricas  # noqa: E402
import scripts_auxiliares.textos as textos  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resultados" / "metricas"
ap = argparse.ArgumentParser()
ap.add_argument("--unidades", nargs="+", default=["corpo", "resumo"], choices=["corpo", "resumo"])
ap.add_argument("--limit", type=int)
ap.add_argument("--sem-lm", action="store_true")
ap.add_argument("--refaz", action="store_true")
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
files = [(l, p) for l in ("portuguese", "english") for p in sorted((ROOT / f"dataset/{l}").glob("*/*/*.json"))]
if a.limit:
    files = files[: a.limit]
(OUT / "config.json").write_text(json.dumps({"versao": metricas.VERSAO, "lm": metricas.LM_NOMES, "janela": metricas.JANELA,
                                             "spacy": metricas.SPACY, "unidades": a.unidades}, indent=1))
for un in a.unidades:
    dest = OUT / f"metricas_{un}.csv"
    feitos = pd.DataFrame() if a.refaz or not dest.exists() else pd.read_csv(dest)
    ja = set(feitos.source_file) if len(feitos) else set()
    rows = feitos.to_dict("records") if len(feitos) else []
    for i, (lang, p) in enumerate(files):
        if p.name in ja:
            continue
        j = json.load(open(p, encoding="utf-8"))
        texto = textos.unidade(j, lang, un)
        ev, ano, _id = p.stem.split("_")
        r = dict(source_file=p.name, event=ev, year=int(ano), article_id=int(_id), language=lang)
        if len(texto.split()) >= 20:
            r.update(metricas.compute(texto, lang, com_lm=not a.sem_lm))
        rows.append(r)
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(dest, index=False)
            print(un, len(rows), "/", len(files), flush=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print("ok", un, len(rows), flush=True)
