"""Gráficos temporais das métricas linguísticas selecionadas (src/config_metricas.py), UM ARQUIVO POR CATEGORIA, inglês (esquerda) e português (direita) lado a lado.
Cada linha de painéis = uma métrica (eixo y compartilhado entre EN e PT na linha, para não esconder diferenças de nível — que, em várias métricas, dependem do idioma/modelo).
Ponto = média ± 1 desvio-padrão amostral dos artigos no ano (corpo); linha tracejada = média de 2020–22; texto = d de Cohen (2023–25 vs 2020–22, desvio combinado).
Cores: EN = slot 1 (azul), PT = slot 2 (laranja) da paleta do skill dataviz, iguais em todas as figuras. Saídas: resultados/analise/graficos_metricas/<categoria>.png e metricas_ano.csv.
Uso: python3 -I src/grafico_metricas.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
_PASTAS_SAIDA = [ROOT / "resultados" / d for d in ("analise", "auditoria", "metricas", "pangram", "canonico", "idioma")]
for _p in _PASTAS_SAIDA:  # o pacote não traz resultados/ (só resultados_referencia/): cria o que faltar
    _p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))


CATEGORIAS = {
    "Complexidade lexical": ["mtld", "hdd", "lexical_density"],
    "Previsibilidade": ["llama_bpc", "llama_surp_sd"],
    "Estilo": ["burstiness", "bigram_rep_per1k", "ai_marker_per1k", "hedge_per1k"],
    "Legibilidade": ["fre"],
    "Sintaxe": ["mdd", "connector_per1k", "sent_len_mean"],
    "Tamanho do texto (controle)": ["n_words", "n_sentences"],
}


OUT = ROOT / "resultados/analise/graficos/metricas"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "resultados/metricas/metricas_corpo.csv")
NOME = {  # rótulo (e leitura da direção)
    "mtld": ("MTLD", "maior = vocabulário mais diverso"), "hdd": ("HD-D", "maior = vocabulário mais diverso"),
    "lexical_density": ("Densidade lexical", "fração de palavras de conteúdo"),
    "llama_bpc": ("Llama bits por caractere", "menor = texto mais previsível"), "llama_surp_sd": ("Llama desvio do surprisal", "variabilidade da surpresa por token"),
    "burstiness": ("Burstiness", "variação (σ/µ) do tamanho de frase"), "bigram_rep_per1k": ("Repetição de bigramas", "por 1.000 bigramas, janelas de 1.000 palavras"),
    "ai_marker_per1k": ("Marcadores de IA", "por 1.000 palavras"), "hedge_per1k": ("Atenuadores", "por 1.000 palavras"),
    "fre": ("Flesch Reading Ease", "maior = mais fácil de ler"),
    "mdd": ("Distância de dependência (MDD)", "média por frase"), "connector_per1k": ("Conectores", "por 1.000 palavras"), "sent_len_mean": ("Palavras por frase", "média"),
    "n_words": ("Palavras (corpo)", ""), "n_sentences": ("Frases (corpo)", ""),
}
COR = {"en": "#2a78d6", "pt": "#eb6834"}
TIT = {"en": "Inglês", "pt": "Português"}
LANG = {"en": "english", "pt": "portuguese"}
SURF, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9992"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
YEARS = list(range(2020, 2026))

g = d.groupby(["language", "year"])
tabela = []
for cat, ms in CATEGORIAS.items():
    ms = [m for m in ms if m in d.columns]
    fig, axes = plt.subplots(len(ms), 2, figsize=(11.5, 2.55 * len(ms) + 1.3), sharex=True, facecolor=SURF, squeeze=False)
    for i, m in enumerate(ms):
        lo, hi = np.inf, -np.inf
        for j, lg in enumerate(("en", "pt")):
            x = d[d.language == LANG[lg]]
            s = x.groupby("year")[m].agg(["mean", "std"]).reindex(YEARS)
            pre, pos = x[x.year <= 2022][m].dropna(), x[x.year >= 2023][m].dropna()
            sd = np.sqrt(((len(pre) - 1) * pre.var() + (len(pos) - 1) * pos.var()) / (len(pre) + len(pos) - 2))
            dc = (pos.mean() - pre.mean()) / sd
            for y, r in s.iterrows():
                tabela.append(dict(categoria=cat, metrica=m, language=lg, year=y, media=r["mean"], dp=r["std"]))
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            ax.axhline(pre.mean(), color=REF, lw=1.2, ls=(0, (4, 3)), zorder=1)
            ax.axvline(2022.5, color="#d2d1cb", lw=1, zorder=0)
            ax.errorbar(YEARS, s["mean"], yerr=s["std"], color=COR[lg], lw=2, marker="o", ms=7, mfc=COR[lg], mec=SURF, mew=1.2, capsize=3.5, capthick=1.3, elinewidth=1.3, zorder=3)
            lo, hi = min(lo, (s["mean"] - s["std"]).min()), max(hi, (s["mean"] + s["std"]).max())
            ax.set_title(f"d = {dc:+.2f}".replace(".", ","), loc="right", fontsize=9.5, color=INK2, pad=4)
            if NOME[m][1]:
                ax.set_title(NOME[m][1], loc="left", fontsize=8.6, color=INK2, pad=4)
            ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#c9c8c2")
            ax.tick_params(length=0)
            ax.set_xticks(YEARS)
            ax.set_xlim(2019.6, 2025.4)
            if i == 0:
                ax.annotate(TIT[lg], xy=(0.5, 1.2), xycoords="axes fraction", ha="center", va="bottom", fontsize=12.5, color=INK)
        pad = 0.06 * (hi - lo)
        for j in range(2):
            axes[i, j].set_ylim(lo - pad, hi + pad)
        axes[i, 0].set_ylabel(NOME[m][0], fontsize=10.5, color=INK)
    fig.suptitle(f"{cat}: média por ano ± 1 desvio-padrão (corpo dos artigos)", x=0.012, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.012, 0.0115, "Linha vertical = nov/2022 (ChatGPT). Tracejado = média de 2020–22. d = tamanho de efeito 2023–25 vs 2020–22 (Cohen). Eixo y igual em inglês e português em cada linha.",
             fontsize=8.8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94), h_pad=1.5, w_pad=1.0)
    arq = cat.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("ã", "a").replace("í", "i").replace("é", "e").replace("ç", "c")
    fig.savefig(OUT / f"{arq}.png", dpi=150, facecolor=SURF)
    plt.close(fig)
pd.DataFrame(tabela).to_csv(ROOT / "resultados/analise/metricas_ano.csv", index=False)
print(sorted(p.name for p in OUT.glob("*.png")))
