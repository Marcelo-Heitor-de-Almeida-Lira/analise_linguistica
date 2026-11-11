"""Comparação PRÉ (2020–22) × PÓS (2023–25) das métricas linguísticas selecionadas (src/config_metricas.py), por idioma, com correção para múltiplos testes.

Teste principal: Mann-Whitney U bilateral (não paramétrico; o esboço prevê Mann-Whitney, α = 0,05) sobre os artigos (corpo).
Tamanhos de efeito: rank-biserial (= delta de Cliff, −1..+1; positivo = pós maior), d de Cohen (desvio combinado) com IC 95% por bootstrap (2.000 reamostragens, seed 0), Δ das medianas.
Família de testes = 13 métricas × 2 idiomas = 26 testes; p ajustado por HOLM (controla a probabilidade de qualquer falso positivo, FWER) e q por BENJAMINI-HOCHBERG (controla a
proporção esperada de falsas descobertas, FDR). Linhas de controle (n_words, n_sentences) e a referência (escore_E do Pangram) são reportadas mas fora da família.
Sensibilidades (cada uma com Holm sobre as 26 métricas × idiomas): (a) sem 2023 (pré × 2024–25; 2023 é ano de transição);
(b) ajuste por conferência: OLS da métrica padronizada (z dentro do idioma) ~ pós + conferência, com erros-padrão robustos HC3 (coef. em desvios-padrão). Diferenças entre conferências
(ex.: artigos de bancos de dados do SBBD, com fórmulas e código SQL, têm mais símbolos e menor previsibilidade) são características das conferências, não ruído, e nenhum artigo é excluído.
"Robusto" = significativo (Holm < 0,05) e com o mesmo sinal na principal e nas duas sensibilidades.
Saídas: resultados/analise/testes_pre_pos.csv e .md.  Uso: python3 -I src/testes_pre_pos.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LANG_DIR = {"pt": "portuguese", "en": "english"}

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts_auxiliares.config_metricas import CATEGORIAS  # noqa: E402

d = pd.read_csv(ROOT / "resultados/metricas/metricas_corpo.csv")
d = d.merge(pd.read_csv(ROOT / "resultados/pangram/pangram_artigos.csv")[["source_file", "escore_E"]], on="source_file")
FAMILIA = [m for c, ms in CATEGORIAS.items() if not c.startswith("Tamanho") for m in ms]
OUTROS = [("n_words", "Tamanho do texto (controle)"), ("n_sentences", "Tamanho do texto (controle)"), ("escore_E", "Referência (Pangram)")]
CAT = {m: c for c, ms in CATEGORIAS.items() for m in ms}
rng = np.random.default_rng(0)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)[::-1]; adj = np.empty(m); run = 1
    for k, i in enumerate(o):
        rank = m - k; run = min(run, p[i] * m / rank); adj[i] = run
    return adj


def efeito(a, b):
    """a = pré, b = pós."""
    u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1
    sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return p, r, (b.mean() - a.mean()) / sd


def d_ic(a, b, B=2000):
    v = []
    for _ in range(B):
        x, y = rng.choice(a, len(a)), rng.choice(b, len(b))
        sd = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2))
        v.append((y.mean() - x.mean()) / sd)
    return np.percentile(v, [2.5, 97.5])


def ols_pos(x, m):
    """métrica padronizada ~ pós + conferência; erros HC3. Retorna (beta_pós, p)."""
    z = (x[m] - x[m].mean()) / x[m].std()
    X = np.column_stack([np.ones(len(x)), (x.year >= 2023).astype(float).values] + [(x.event == e).astype(float).values for e in sorted(x.event.unique())[1:]])
    ok = ~z.isna().values
    X, y = X[ok], z.values[ok]
    XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ X.T @ y; e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = (X * (e / (1 - h))[:, None] ** 2).T @ X
    cov = XtXi @ meat @ XtXi
    t = beta[1] / np.sqrt(cov[1, 1])
    return beta[1], 2 * stats.t.sf(abs(t), len(y) - X.shape[1])


linhas = []
for lg in ("en", "pt"):
    x = d[d.language == LANG_DIR[lg]]
    pre_m, pos_m = x.year <= 2022, x.year >= 2023
    for m, cat in [(m, CAT[m]) for m in FAMILIA] + OUTROS:
        a, b = x.loc[pre_m, m].dropna(), x.loc[pos_m, m].dropna()
        p, r, dc = efeito(a, b)
        lo, hi = d_ic(a.values, b.values)
        # sensibilidades
        a2, b2 = x.loc[x.year <= 2022, m].dropna(), x.loc[x.year >= 2024, m].dropna()
        p_a, r_a, _ = efeito(a2, b2)
        beta, p_c = ols_pos(x, m)
        linhas.append(dict(language=lg, categoria=cat, metrica=m, familia=m in FAMILIA, n_pre=len(a), n_pos=len(b), med_pre=a.median(), med_pos=b.median(), delta_med=b.median() - a.median(),
                           media_pre=a.mean(), media_pos=b.mean(), p=p, rank_biserial=r, d=dc, d_ic_lo=lo, d_ic_hi=hi,
                           p_sem2023=p_a, r_sem2023=r_a, beta_ajustado=beta, p_ajustado=p_c))
t = pd.DataFrame(linhas)
f = t.familia
for col, novo in (("p", "p_holm"), ("p_sem2023", "p_holm_sem2023"), ("p_ajustado", "p_holm_ajustado")):
    t.loc[f, novo] = holm(t.loc[f, col])
t.loc[f, "q_bh"] = bh(t.loc[f, "p"])
sinal = lambda a, b: np.sign(a) == np.sign(b)
t["robusto"] = (t.p_holm < 0.05) & (t.p_holm_sem2023 < 0.05) & (t.p_holm_ajustado < 0.05) & sinal(t.rank_biserial, t.r_sem2023) & sinal(t.rank_biserial, t.beta_ajustado)
t["robusto"] = t["robusto"].astype(object); t.loc[~f, ["p_holm", "q_bh", "robusto"]] = [np.nan, np.nan, np.nan]
t.to_csv(ROOT / "resultados/analise/testes_pre_pos.csv", index=False)


def estrelas(p):
    return "" if pd.isna(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


fmt = lambda v, n=2: ("%+.*f" % (n, v)).replace(".", ",")
md = ["| Categoria | Métrica | EN: Δ mediana | r | d [IC 95%] | p Holm | q BH | robusto | PT: Δ mediana | r | d [IC 95%] | p Holm | q BH | robusto |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
ult = None
for m, cat in [(m, CAT[m]) for m in FAMILIA] + OUTROS:
    cols = []
    for lg in ("en", "pt"):
        r = t[(t.language == lg) & (t.metrica == m)].iloc[0]
        pH = "—" if pd.isna(r.p_holm) else ("<0,001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}".replace(".", ",")) + " " + estrelas(r.p_holm)
        qB = "—" if pd.isna(r.q_bh) else ("<0,001" if r.q_bh < 0.001 else f"{r.q_bh:.3f}".replace(".", ","))
        rb = "—" if pd.isna(r.robusto) else ("sim" if r.robusto else "não")
        cols += [fmt(r.delta_med, 3), fmt(r.rank_biserial), f"{fmt(r.d)} [{fmt(r.d_ic_lo)}; {fmt(r.d_ic_hi)}]", pH, qB, rb]
    md.append(f"| {cat if cat != ult else ''} | {m} | " + " | ".join(cols) + " |")
    ult = cat
(ROOT / "resultados/analise/testes_pre_pos.md").write_text("\n".join(md) + "\n")
fam = t[t.familia]
print("\n".join(md))
for lg, g in fam.groupby("language"):
    print(f"\n{lg}: significativos (p bruto<0,05) {int((g.p<0.05).sum())}/13 | Holm<0,05 {int((g.p_holm<0.05).sum())}/13 | BH q<0,05 {int((g.q_bh<0.05).sum())}/13 | robustos {int(g.robusto.sum())}/13")
print(f"\nP(≥1 falso positivo) com 26 testes independentes a α=0,05: {100*(1-0.95**26):.0f}%")
