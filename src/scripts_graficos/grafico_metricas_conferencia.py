"""Gráficos temporais das métricas linguísticas selecionadas, separados por CONFERÊNCIA: um arquivo por categoria (src/config_metricas.py), inglês (esquerda) e português (direita),
uma linha de painéis por métrica, uma linha por conferência (mesmas cores/marcadores do grafico_conferencias.py). Ponto = média dos artigos da conferência no ano (sem barras de desvio,
para não poluir seis séries por painel; o desvio está em grafico_metricas.py); marcador vazado = célula com n < 10 artigos. Tracejado cinza = média do idioma no ano (todas as conferências).
Eixo y comum em inglês e português na linha. Saídas: resultados/analise/graficos_metricas_conferencia/<categoria>.png e metricas_conferencia_ano.csv.
Uso: python3 -I src/grafico_metricas_conferencia.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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


OUT = ROOT / "resultados/analise/graficos/metricas_conferencia"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "resultados/metricas/metricas_corpo.csv")
NOME = {
    "mtld": ("MTLD", "maior = vocabulário mais diverso"), "hdd": ("HD-D", "maior = vocabulário mais diverso"), "lexical_density": ("Densidade lexical", "fração de palavras de conteúdo"),
    "llama_bpc": ("Llama bits por caractere", "menor = texto mais previsível"), "llama_surp_sd": ("Llama desvio do surprisal", "variabilidade da surpresa por token"),
    "burstiness": ("Burstiness", "variação (σ/µ) do tamanho de frase"), "bigram_rep_per1k": ("Repetição de bigramas", "por 1.000 bigramas"),
    "ai_marker_per1k": ("Marcadores de IA", "por 1.000 palavras"), "hedge_per1k": ("Atenuadores", "por 1.000 palavras"),
    "fre": ("Flesch Reading Ease", "maior = mais fácil de ler"), "mdd": ("Distância de dependência", "média por frase"),
    "connector_per1k": ("Conectores", "por 1.000 palavras"), "sent_len_mean": ("Palavras por frase", "média"), "n_words": ("Palavras (corpo)", ""), "n_sentences": ("Frases (corpo)", ""),
}
EVENTS = ["brasnam", "eniac", "sbbd", "sbrc", "semish", "wics"]
EV = {"brasnam": "BraSNAM", "eniac": "ENIAC", "sbbd": "SBBD", "sbrc": "SBRC", "semish": "SEMISH", "wics": "WICS"}
COR = dict(zip(EVENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARK = dict(zip(EVENTS, ["o", "s", "^", "D", "v", "P"]))
TIT = {"en": "Inglês", "pt": "Português"}
LANG = {"en": "english", "pt": "portuguese"}
SURF, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9992"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
YEARS = list(range(2020, 2026))
tabela = []
for cat, ms in CATEGORIAS.items():
    ms = [m for m in ms if m in d.columns]
    fig, axes = plt.subplots(len(ms), 2, figsize=(12, 2.7 * len(ms) + 1.6), sharex=True, facecolor=SURF, squeeze=False)
    for i, m in enumerate(ms):
        lo, hi = np.inf, -np.inf
        for j, lg in enumerate(("en", "pt")):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            x = d[d.language == LANG[lg]]
            ax.plot(YEARS, x.groupby("year")[m].mean().reindex(YEARS), color=REF, lw=1.4, ls=(0, (4, 3)), zorder=1)
            ax.axvline(2022.5, color="#d2d1cb", lw=1, zorder=0)
            for ev in EVENTS:
                s = x[x.event == ev].groupby("year")[m].agg(media="mean", n="count").reindex(YEARS).dropna()
                ax.plot(s.index, s.media, color=COR[ev], lw=1.8, zorder=2)
                for y, r in s.iterrows():
                    ax.plot(y, r.media, marker=MARK[ev], ms=7, mfc=COR[ev] if r.n >= 10 else SURF, mec=COR[ev], mew=1.6, zorder=3, ls="none")
                    tabela.append(dict(metrica=m, language=lg, event=ev, year=y, media=r.media, n=int(r.n)))
                lo, hi = min(lo, s.media.min()), max(hi, s.media.max())
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
    fig.suptitle(f"{cat}: média por conferência e ano (corpo dos artigos)", x=0.012, ha="left", fontsize=14, fontweight="bold", color=INK)
    h = [Line2D([0], [0], color=COR[e], marker=MARK[e], ms=7, lw=1.8, label=EV[e]) for e in EVENTS]
    h += [Line2D([0], [0], color=REF, lw=1.4, ls=(0, (4, 3)), label="média do idioma"), Line2D([0], [0], color=INK2, marker="o", mfc=SURF, mec=INK2, mew=1.6, ms=7, lw=0, label="n < 10 artigos")]
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False, fontsize=9.3, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 0.945), h_pad=1.5, w_pad=1.0)
    arq = cat.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("ã", "a").replace("í", "i").replace("é", "e").replace("ç", "c")
    fig.savefig(OUT / f"{arq}.png", dpi=140, facecolor=SURF)
    plt.close(fig)
pd.DataFrame(tabela).to_csv(ROOT / "resultados/analise/metricas_conferencia_ano.csv", index=False)
print(sorted(p.name for p in OUT.glob("*.png")))
