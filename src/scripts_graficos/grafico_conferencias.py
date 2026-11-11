"""Escore de IA do Pangram-Llama (escore_E, 0 = humano ... 1 = IA; média do corpo de cada artigo) por conferência e ano.
Pequenos múltiplos: uma linha de painéis por idioma (EN em cima, PT embaixo), um painel por conferência; eixos y compartilhados.
Em cada ponto: média ± 1 desvio-padrão amostral (ddof=1) dos ARTIGOS daquela célula (não é erro-padrão). Marcador vazado = n < 10 (n não é mostrado no gráfico, só no CSV); sem barra quando n = 1.
Linha tracejada cinza = média de todas as conferências no idioma. Cores: slots 1-6 da paleta do skill dataviz (ordem fixa por conferência).
Saídas: resultados/analise/escore_pangram_conferencias_ano.{png,csv}.  Uso: python3 -I src/grafico_conferencias.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent
_PASTAS_SAIDA = [ROOT / "resultados" / d for d in ("analise", "auditoria", "metricas", "pangram", "canonico", "idioma")]
for _p in _PASTAS_SAIDA:  # o pacote não traz resultados/ (só resultados_referencia/): cria o que faltar
    _p.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "resultados/analise/graficos"
OUT.mkdir(parents=True, exist_ok=True)
a = pd.read_csv(ROOT / "resultados/pangram/pangram_artigos.csv")
g = a.groupby(["language", "event", "year"]).escore_E.agg(media="mean", dp="std", n="size").reset_index()
g.to_csv(OUT / "escore_pangram_conferencias_ano.csv", index=False)
todos = a.groupby(["language", "year"]).escore_E.mean().reset_index()

EVENTS = ["brasnam", "eniac", "sbbd", "sbrc", "semish", "wics"]
NOME = {"brasnam": "BraSNAM", "eniac": "ENIAC", "sbbd": "SBBD", "sbrc": "SBRC", "semish": "SEMISH", "wics": "WICS"}
COR = dict(zip(EVENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARK = dict(zip(EVENTS, ["o", "s", "^", "D", "v", "P"]))
SURF, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9992"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

fig, axes = plt.subplots(2, 6, figsize=(15, 7.4), sharey=True, sharex=True, facecolor=SURF)
for i, (lg, titulo) in enumerate([("english", "Inglês"), ("portuguese", "Português")]):
    ref = todos[todos.language == lg]
    for j, ev in enumerate(EVENTS):
        ax = axes[i, j]
        ax.set_facecolor(SURF)
        ax.plot(ref.year, ref.escore_E, color=REF, lw=1.4, ls=(0, (4, 3)), zorder=1)
        s = g[(g.language == lg) & (g.event == ev)].sort_values("year")
        ax.plot(s.year, s.media, color=COR[ev], lw=2, zorder=2)
        for _, r in s.iterrows():
            if r.n > 1:
                ax.errorbar(r.year, r.media, yerr=r.dp, color=COR[ev], lw=1.4, capsize=3.5, capthick=1.4, zorder=2, alpha=0.75)
            ax.plot(r.year, r.media, marker=MARK[ev], ms=8.5, mfc=COR[ev] if r.n >= 10 else SURF, mec=COR[ev], mew=1.8, zorder=3, ls="none")
        ax.axvline(2022.5, color="#d2d1cb", lw=1, zorder=0)
        if i == 0:
            ax.set_title(NOME[ev], fontsize=12.5, color=INK, pad=8)
        ax.set_xlim(2019.5, 2025.5)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks(range(2020, 2026, 1))
        ax.set_xticklabels(["20", "21", "22", "23", "24", "25"], fontsize=9)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        if j == 0:
            ax.set_ylabel(titulo, fontsize=11.5, color=INK, labelpad=10)
        else:
            ax.tick_params(labelleft=False)
fig.text(0.012, 0.965, "Escore de IA do Pangram-Llama por conferência e ano (média ± 1 desvio-padrão dos artigos)", fontsize=15, fontweight="bold", color=INK, ha="left", va="center")
fig.text(0.012, 0.935, "Escore: 0 = humano … 1 = IA, média do corpo de cada artigo. Linha vertical = nov/2022 (ChatGPT).", fontsize=9.5, color=INK2, ha="left", va="center")
h = [Line2D([0], [0], color=REF, lw=1.4, ls=(0, (4, 3)), label="média de todas as conferências do idioma"),
     Line2D([0], [0], color=INK2, marker="o", mfc=SURF, mec=INK2, mew=1.8, ms=8.5, lw=0, label="n < 10 artigos"),
     Line2D([0], [0], color=INK2, lw=1.4, marker="|", ms=12, label="± 1 desvio-padrão (sem barra se n = 1)")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.045, 1, 0.915), h_pad=1.6, w_pad=0.8)
fig.savefig(OUT / "escore_pangram_conferencias_ano.png", dpi=160, facecolor=SURF)
print(g.round(2).to_string(index=False))
