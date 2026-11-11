"""Pangram-Llama sobre o RESUMO (abstract no idioma do artigo), mesmo método de calcula_pangram.py: janelas sem sobreposição de 512 tokens
(janelas < 32 tokens descartadas); como o resumo é curto, normalmente 1 janela. Artigos sem resumo no idioma do artigo ficam de fora (31).
Saída: resultados/pangram/pangram_resumo.csv (por artigo: n_tokens, n_janelas, P0..P3, escore_E, classe). Uso: python3 src/calcula_pangram_resumo.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scripts_auxiliares.pangram as pangram  # noqa: E402
import scripts_auxiliares.textos as textos  # noqa: E402

JANELA, MIN_TOK, BATCH = 512, 32, 8
LANG_DIR = {"portuguese": "pt", "english": "en"}
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resultados" / "pangram"
OUT.mkdir(parents=True, exist_ok=True)
tok, _ = pangram.load()
rows = []
for language in ("portuguese", "english"):
    lang = LANG_DIR.get(language)
    for p in sorted((ROOT / f"dataset/{language}").glob("*/*/*.json")):
        t = textos.resumo(json.load(open(p, encoding="utf-8")), lang)
        ids = tok(t, add_special_tokens=False)["input_ids"] if t else []
        jan = [(tok.decode(ids[k:k + JANELA]), len(ids[k:k + JANELA])) for k in range(0, len(ids), JANELA) if len(ids[k:k + JANELA]) >= MIN_TOK]
        if not jan:
            continue
        pr = np.array(pangram.probs([j[0] for j in jan], batch=BATCH, max_length=JANELA + 4))
        w = np.array([j[1] for j in jan])
        m = np.average(pr, axis=0, weights=w)
        ev, ano, _id = p.stem.split("_")
        rows.append(dict(source_file=p.name, event=ev, year=int(ano), article_id=int(_id), language=language, categoria="Resumo", n_tokens=int(w.sum()), n_janelas=len(jan),
                         P0=m[0], P1=m[1], P2=m[2], P3=m[3], escore_E=(m[1] + 2 * m[2] + 3 * m[3]) / 300, classe=int(np.argmax(m))))
        if len(rows) % 200 == 0:
            print(len(rows), flush=True)
pd.DataFrame(rows).to_csv(OUT / "pangram_resumo.csv", index=False)
print("ok", len(rows))
