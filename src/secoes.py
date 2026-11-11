"""Mapeamento dos títulos de seção do parser (`secao_titulo`) para 5 categorias do artigo + agregação do Pangram por artigo × categoria.

Categorias: Resumo (abstract no idioma do artigo, pontuado à parte), Introdução, Trabalhos relacionados (inclui referencial/fundamentação teórica, background), Metodologia (inclui proposta/abordagem/solução proposta, setup, dados),
Resultados (inclui avaliação, experimentos, análise, discussão, estudo de caso), Conclusão (inclui considerações finais e trabalhos futuros). O resto = "Outras" (fora dos gráficos).
Regras por expressão regular em PT e EN, aplicadas na ORDEM abaixo ao título normalizado (sem numeração). "Preamble" (texto antes do 1º título) = Introdução.
Seções do parser rotuladas "Referências/References" que não eram bibliografia caem em "Outras" (o conteúdo é texto do artigo de posição desconhecida).
O mapeamento é uma decisão metodológica provisória: resultados/analise/mapeamento_secoes.csv lista cada título distinto, sua frequência e a categoria atribuída (revisar).
Saídas: resultados/pangram/pangram_artigo_categoria.csv (P0..P3, escore_E, classe por artigo × categoria) e o mapeamento acima.
Uso: python3 -I src/secoes.py"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resultados" / "pangram"
OUT.mkdir(parents=True, exist_ok=True)
REGRAS = [
    ("Conclusão", r"conclus|considera[cç][õo]es finais|final consideration|concluding|final remarks|trabalhos? futuros?|future work|limita[cç][õo]es e trabalhos"),
    ("Introdução", r"introdu|preamble|^motiva[cç]"),
    ("Trabalhos relacionados", r"related|relacionad|correlat|similares|background|fundamenta|referencial|fundamentos|conceitos|revis[ãa]o (d|bibliogr|sistem|liter)|literature review|systematic (literature )?review|state of the art|estado da arte|preliminar|trabalhos anteriores|base te[óo]rica|theoretical"),
    ("Metodologia", r"metodolog|methodolog|m[ée]todo|\bmethods?\b|materia(l|is) (e|and)|setup|configura[cç][ãa]o experimental|base de dados|conjunto de dados|\bdatasets?\b|defini[cç][ãa]o do problema|problem (formulation|definition|statement)|prova de conceito|proof of concept|desenvolvimento|development"),
    ("Resultados", r"result|avalia|evaluat|experiment|an[áa]lis|analys|desempenho|performance|discuss|estudos? de caso|case stud|valida[cç]|validation|caso de uso|use case"),
    ("Metodologia", r"propost|proposal|\bproposed\b|solu[cç][ãa]o|solution|abordagem|approach|algoritmo|algorithm|modelo|model\b|arquitetura|architecture|implementa|framework|sistema"),
]
CATEGORIAS = ["Resumo", "Introdução", "Trabalhos relacionados", "Metodologia", "Resultados", "Conclusão"]
norm = lambda t: re.sub(r"^\W*(\d+(\.\d+)*\.?|[ivx]+\.)?\s*", "", str(t).lower()).strip(" .:")


def categoria(titulo):
    t = norm(titulo)
    for cat, rx in REGRAS:
        if re.search(rx, t):
            return cat
    return "Outras"


if __name__ == "__main__":
    w = pd.read_csv(ROOT / "resultados/pangram/pangram_janelas.csv")
    w["categoria"] = w.secao_titulo.map(categoria)
    # mapeamento (por título normalizado)
    m = w.assign(t=w.secao_titulo.map(norm)).groupby(["t", "categoria"]).agg(secoes=("secao_idx", "size"), tokens=("n_tokens", "sum")).reset_index().sort_values("tokens", ascending=False)
    m.to_csv(ROOT / "resultados/analise/mapeamento_secoes.csv", index=False)
    # escore por janela (E = (p1 + 2 p2 + 3 p3) / 300) e agregação por artigo × categoria, ponderada por tokens
    w["E"] = (w.p1 + 2 * w.p2 + 3 * w.p3) / 300
    rows = []
    for (f, c), g in w.groupby(["source_file", "categoria"]):
        pr = np.average(g[["p0", "p1", "p2", "p3"]].values, axis=0, weights=g.n_tokens)
        rows.append(dict(source_file=f, categoria=c, n_tokens=int(g.n_tokens.sum()), n_janelas=len(g), P0=pr[0], P1=pr[1], P2=pr[2], P3=pr[3],
                         escore_E=np.average(g.E, weights=g.n_tokens), classe=int(np.argmax(pr))))
    d = pd.DataFrame(rows)
    art = pd.read_csv(ROOT / "resultados/pangram/pangram_artigos.csv")[["source_file", "event", "year", "article_id", "language"]]
    d = art.merge(d, on="source_file")
    rp = ROOT / "resultados/pangram/pangram_resumo.csv"
    if rp.exists():  # resumo (abstract no idioma do artigo) entra como categoria "Resumo" (src/calcula_pangram_resumo.py)
        d = pd.concat([d, pd.read_csv(rp)[d.columns]], ignore_index=True)
    d.to_csv(OUT / "pangram_artigo_categoria.csv", index=False)
    tot = w.n_tokens.sum()
    print("Cobertura (% dos tokens) por categoria:\n", (w.groupby("categoria").n_tokens.sum() / tot * 100).round(1).sort_values(ascending=False).to_string())
    cob = w.assign(mapeado=w.categoria != "Outras").merge(art[["source_file", "year", "language"]], on="source_file")
    print("\n% de tokens classificados por idioma e período:\n", cob.assign(per=np.where(cob.year <= 2022, "2020-22", "2023-25")).groupby(["language", "per"]).apply(lambda g: round(100 * g.loc[g.mapeado, "n_tokens"].sum() / g.n_tokens.sum(), 1), include_groups=False).to_string())
    print("\nartigos com cada categoria (de %d):" % art.shape[0], d.groupby("categoria").source_file.nunique().to_dict())
    print("\nmaiores títulos em 'Outras':\n", m[m.categoria == "Outras"].head(25)[["t", "secoes", "tokens"]].to_string(index=False))
