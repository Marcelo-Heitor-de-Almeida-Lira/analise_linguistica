"""Métricas linguísticas ao longo dos anos POR CONFERÊNCIA, COM desvio-padrão em cada ponto: um arquivo por categoria e idioma (src/config_metricas.py).
Pequenos múltiplos: uma linha de painéis por métrica, uma coluna por conferência (eixo y compartilhado na linha, para comparar as conferências); ponto = média ± 1 desvio-padrão
amostral dos artigos (corpo); sem barra se n = 1; marcador vazado = n < 10. Tracejado = média do idioma no ano. Cores/marcadores das conferências iguais aos demais gráficos.
Saídas: resultados/analise/graficos_metricas_conferencia_dp/<categoria>_<idioma>.png.   Uso: python3 -I src/grafico_metricas_conferencia_dp.py"""
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

OUT = ROOT / "resultados/analise/graficos/metricas_conferencia_dp"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "resultados/metricas/metricas_corpo.csv")
NOME = {"mtld": "MTLD", "hdd": "HD-D", "lexical_density": "Densidade lexical", "llama_bpc": "Llama bits/caractere", "llama_surp_sd": "Llama desvio surprisal",
        "burstiness": "Burstiness", "bigram_rep_per1k": "Repetição de bigramas", "ai_marker_per1k": "Marcadores de IA", "hedge_per1k": "Atenuadores",
        "fre": "Flesch Reading Ease", "mdd": "MDD", "connector_per1k": "Conectores", "sent_len_mean": "Palavras por frase"}
EVENTS = ["brasnam", "eniac", "sbbd", "sbrc", "semish", "wics"]
EV = {"brasnam": "BraSNAM", "eniac": "ENIAC", "sbbd": "SBBD", "sbrc": "SBRC", "semish": "SEMISH", "wics": "WICS"}
COR = dict(zip(EVENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARK = dict(zip(EVENTS, ["o", "s", "^", "D", "v", "P"]))
TIT = {"en": "inglês", "pt": "português"}
LANG = {"en": "english", "pt": "portuguese"}
SURF, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9992"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
YEARS = list(range(2020, 2026))
for cat, ms in CATEGORIAS.items():
    if cat.startswith("Tamanho"):
        continue
    for lg in ("en", "pt"):
        x = d[d.language == LANG[lg]]
        fig, axes = plt.subplots(len(ms), 6, figsize=(14.5, 2.15 * len(ms) + 1.2), sharex=True, sharey="row", facecolor=SURF, squeeze=False)
        for i, m in enumerate(ms):
            ref = x.groupby("year")[m].mean().reindex(YEARS)
            for j, ev in enumerate(EVENTS):
                ax = axes[i, j]
                ax.set_facecolor(SURF)
                ax.plot(YEARS, ref, color=REF, lw=1.3, ls=(0, (4, 3)), zorder=1)
                ax.axvline(2022.5, color="#d2d1cb", lw=1, zorder=0)
                s = x[x.event == ev].groupby("year")[m].agg(media="mean", dp="std", n="count").reindex(YEARS).dropna(subset=["media"])
                ax.plot(s.index, s.media, color=COR[ev], lw=1.8, zorder=2)
                for y, r in s.iterrows():
                    if r.n > 1:
                        ax.errorbar(y, r.media, yerr=r.dp, color=COR[ev], lw=1.2, capsize=2.8, capthick=1.2, alpha=0.7, zorder=2)
                    ax.plot(y, r.media, marker=MARK[ev], ms=6.5, mfc=COR[ev] if r.n >= 10 else SURF, mec=COR[ev], mew=1.5, zorder=3, ls="none")
                ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
                ax.set_axisbelow(True)
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color("#c9c8c2")
                ax.tick_params(length=0, labelsize=8)
                ax.set_xticks(YEARS)
                ax.set_xticklabels(["20", "21", "22", "23", "24", "25"])
                ax.set_xlim(2019.5, 2025.5)
                if i == 0:
                    ax.set_title(EV[ev], fontsize=11.5, color=INK, pad=6)
                if j == 0:
                    ax.set_ylabel(NOME[m], fontsize=9.5, color=INK)
                else:
                    ax.tick_params(labelleft=False)
        fig.suptitle(f"{cat} em {TIT[lg]}: média ± 1 desvio-padrão por conferência e ano (corpo)", x=0.012, ha="left", fontsize=13.5, fontweight="bold", color=INK)
        fig.text(0.012, 0.004, "Linha vertical = nov/2022 (ChatGPT). Tracejado = média do idioma. Marcador vazado = menos de 10 artigos na célula; sem barra quando há 1 artigo. Eixo y comum a todas as conferências em cada linha.",
                 fontsize=8.6, color=INK2, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.025, 1, 0.945), h_pad=1.0, w_pad=0.6)
        arq = cat.lower().replace(" ", "_").replace("ã", "a").replace("í", "i").replace("é", "e").replace("ç", "c")
        fig.savefig(OUT / f"{arq}_{lg}.png", dpi=125, facecolor=SURF)
        plt.close(fig)
print(sorted(p.name for p in OUT.glob("*.png")))
