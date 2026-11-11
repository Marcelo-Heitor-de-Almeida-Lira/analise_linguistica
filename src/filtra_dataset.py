"""Gera dataset_pt/ e dataset_en/ a partir dos JSONs estruturados de dataset/converted_json/ (a entrada real das análises).

Entrada : dataset/converted_json/*.json  (chaves: metadata, abstract{pt,en}, sections[], full_text)
Saída   : dataset/<lang>/<evento>/<ano>/<arquivo>.json  (cópia integral: mantém abstract e sections),
          resultados/canonico/lista_canonica.csv e excluidos.csv.
Passos  : (1) idioma por langdetect sobre o corpo (seções) — o full_text pode começar com abstract PT+EN;
          (2) exclui o que não é artigo/está fora do escopo (EXCLUIR, revisão manual, ver research.md §4e);
          (3) checa duplicatas (texto idêntico, título idêntico, similaridade TF-IDF > 0,8) e mantém o menor id;
          (4) marca `curto` (full_text < 10.000 caracteres) sem excluir.
Uso     : python3 src/gera_datasets.py   (sem -I: langdetect/sklearn estão no site-packages do usuário)
"""
import hashlib
import json
import re
from pathlib import Path

import shutil
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect_langs
from sklearn.feature_extraction.text import TfidfVectorizer

DetectorFactory.seed = 0
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dataset" / "converted_json"
OUT = ROOT / "resultados" / "canonico"
OUT.mkdir(parents=True, exist_ok=True)
LANG = {"pt": "portuguese", "en": "english", "outros": "other"}

EXCLUIR = {  # revisão manual
    "semish_2025_36704": "Carta de introdução da conferência, não configura artigo",
    "eniac_2024_33567": "Carta de introdução da conferência, não configura artigo",
    "sbbd_2020_13502": "Carta de introdução da conferência, não configura artigo",
    "sbbd_2021_18138": "Carta de introdução da conferência, não configura artigo",
    "sbbd_2023_25381": "Carta de introdução da conferência, não configura artigo",
    "sbbd_2024_30480": "Carta de introdução da conferência, não configura artigo",
    "sbbd_2025_37098": "Carta de introdução da conferência, não configura artigo",
    "sbbd_2020_13477": "sem texto útil (só eixos/legendas de figura, sem título/autores)",
    "eniac_2021_18258": "front matter (visão geral da competição KDD-BR 2021)",
    "eniac_2021_18259": "trabalho curto (KDD-BR 2021)",
    "eniac_2021_18260": "trabalho curto (KDD-BR 2021)",
    "eniac_2021_18261": "trabalho curto (KDD-BR 2021)",
}
norm = lambda t: re.sub(r"\s+", " ", t or "").strip()


def body(j):
    out = []

    def rec(s):
        out.append(s.get("sectionContent") or "")
        for x in s.get("subsections") or []:
            rec(x)

    for s in j["sections"]:
        rec(s)
    return norm(" ".join(out))


def lang_of(text):
    if len(text) < 200:
        return "outros", 0.0
    try:
        top = detect_langs(text)[0]
    except LangDetectException:
        return "outros", 0.0
    return (top.lang if top.lang in ("pt", "en") else "outros"), round(top.prob, 4)


rows, texts = [], []
for p in sorted(SRC.glob("*.json")):
    j = json.load(open(p, encoding="utf-8"))
    ft = norm(j["full_text"])
    lb, pb = lang_of(body(j))
    lf, pf = lang_of(ft)
    ev, ano, _id = p.stem.split("_")
    rows.append(dict(source_file=p.name, event=ev, year=int(ano), id=int(_id), lang=lb, prob=pb, lang_fulltext=lf,
                     title=norm(j["metadata"].get("title")), n_chars=len(ft), nsec=len(j["sections"]),
                     refs=j["metadata"].get("reference_count"), h=hashlib.md5(ft.encode()).hexdigest()))
    texts.append(ft)
d = pd.DataFrame(rows)
d["curto"] = d.n_chars < 10000
d["motivo"] = d.source_file.str[:-5].map(EXCLUIR).fillna("")
d.loc[(d.motivo == "") & (d.lang == "outros"), "motivo"] = "idioma fora de pt/en"

# duplicatas (apenas informativo + remoção): texto idêntico, título idêntico, TF-IDF > 0,8
X = TfidfVectorizer(max_features=50000, sublinear_tf=True).fit_transform(texts)
S = (X @ X.T).tocoo()
pares = {(i, j) for i, j, v in zip(S.row, S.col, S.data) if i < j and v > 0.8}
t = d.title.str.lower().str.replace(r"\W+", "", regex=True)
for _, g in d[(t != "") & t.duplicated(keep=False)].groupby(t):
    idx = list(g.index); pares |= {(a, b) for a in idx for b in idx if a < b}
for _, g in d[d.h.duplicated(keep=False)].groupby("h"):
    idx = list(g.index); pares |= {(a, b) for a in idx for b in idx if a < b}
print("pares duplicados encontrados:", len(pares))
for a, b in sorted(pares):
    drop = b if d.id[a] < d.id[b] else a
    d.loc[drop, "motivo"] = d.loc[drop, "motivo"] or "duplicata (mantido menor id)"

can = d[d.motivo == ""]
can.drop(columns=["motivo", "h"]).to_csv(OUT / "lista_canonica.csv", index=False)
d[d.motivo != ""].drop(columns=["h"]).to_csv(OUT / "excluidos.csv", index=False)

for name in ("dataset_pt", "dataset_en"):
    shutil.rmtree(ROOT / name, ignore_errors=True)
for r in can.itertuples():
    dest = ROOT / "dataset" / LANG[r.lang] / r.event / str(r.year)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / r.source_file, dest / r.source_file)

print("total:", len(d), "| canônicos:", len(can), "| excluídos:", d.motivo.ne("").sum())
print(d[d.motivo != ""][["source_file", "motivo"]].to_string(index=False))
print("idioma corpo vs full_text divergem:", int((can.lang != can.lang_fulltext).sum()))
print(pd.crosstab(can.event, [can.year >= 2023, can.lang]).rename(columns={False: "2020-22", True: "2023-25"}))
print("curtos (<10.000 caracteres) mantidos:", int(can.curto.sum()))
print("prob < 0.99:", int((can.prob < 0.99).sum()))
