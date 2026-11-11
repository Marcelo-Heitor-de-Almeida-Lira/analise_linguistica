"""Gráficos alternativos do Pangram-Llama nos extremos da escala: P0 (classe 0 = humano, %) e P3 (classe 3 = IA, %), por seção e ano, um arquivo por idioma e tipo:
  linhas   : x = ano, uma linha por seção
  barras_ano   : x = ano, uma barra por seção
  barras_secao : x = seção, uma barra por ano (rampa sequencial de um tom)
Cada figura tem 2 painéis (P0 à esquerda, P3 à direita). Valor = média, entre os artigos, da probabilidade média da seção (corpo em janelas de 512 tokens; Resumo = abstract).
Seções: Resumo + 5 categorias (src/secoes.py). Saídas: resultados/analise/graficos_labels/*.png e resultados/analise/labels_secoes_ano.csv.
Uso: python3 -I src/grafico_labels.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent.parent
_PASTAS_SAIDA = [ROOT / "resultados" / d for d in ("analise", "auditoria", "metricas", "pangram", "canonico", "idioma")]
for _p in _PASTAS_SAIDA:  # o pacote não traz resultados/ (só resultados_referencia/): cria o que faltar
    _p.mkdir(parents=True, exist_ok=True)

OUT = ROOT / "resultados/analise/graficos/labels"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "resultados/pangram/pangram_artigo_categoria.csv")
CATS = ["Resumo", "Introdução", "Trabalhos relacionados", "Metodologia", "Resultados", "Conclusão"]
d = d[d.categoria.isin(CATS)]
m = d.groupby(["language", "categoria", "year"])[["P0", "P3"]].mean().reset_index()
print(m.head(15))
m.to_csv(ROOT / "resultados/analise/labels_secoes_ano.csv", index=False)

COR = {"Resumo": "#008300", "Introdução": "#2a78d6", "Trabalhos relacionados": "#eb6834", "Metodologia": "#1baf7a", "Resultados": "#eda100", "Conclusão": "#e87ba4"}
MARK = {"Resumo": "P", "Introdução": "o", "Trabalhos relacionados": "s", "Metodologia": "^", "Resultados": "D", "Conclusão": "v"}
YEARS = list(range(2020, 2026))
RAMP = ["#d6e5f8", "#aecbf1", "#82ade6", "#5590d9", "#2a6bbf", "#173f78"]  # sequencial, um tom (azul), claro -> escuro
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NOME = {"en": "inglês", "pt": "português"}
LANG = {"en": "english", "pt": "portuguese"}
LAB = {"P0": "Classe 0 — humano (%)", "P3": "Classe 3 — gerado por IA (%)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def estilo(ax, ylab):
    ax.set_facecolor(SURF)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(length=0)
    ax.set_title(ylab, loc="left", fontsize=12, color=INK, pad=8)


def val(lg, cat, y, col):
    r = m[(m.language == lg) & (m.categoria == cat) & (m.year == y)]
    return r[col].iloc[0] if len(r) else np.nan


for lg in ("en", "pt"):
    for tipo in ("linhas", "barras_ano", "barras_secao"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=SURF)
        for ax, col in zip(axes, ("P0", "P3")):
            estilo(ax, LAB[col])
            if tipo == "linhas":
                for c in CATS:
                    ys = [val(LANG[lg], c, y, col) for y in YEARS]
                    ax.plot(YEARS, ys, color=COR[c], lw=2, marker=MARK[c], ms=7.5, mfc=COR[c], mec=SURF, mew=1.2, zorder=3)
                ax.set_xticks(YEARS)
                ax.set_xlabel("Ano", fontsize=10)
                ax.axvline(2022.5, color="#d2d1cb", lw=1, zorder=0)
            elif tipo == "barras_ano":
                w = 0.13
                for k, c in enumerate(CATS):
                    ax.bar(np.arange(6) + (k - 2.5) * w, [val(LANG[lg], c, y, col) for y in YEARS], w, color=COR[c], edgecolor=SURF, linewidth=1.2, zorder=3)
                ax.set_xticks(range(6))
                ax.set_xticklabels(YEARS)
                ax.set_xlabel("Ano", fontsize=10)
            else:
                w = 0.13
                for k, y in enumerate(YEARS):
                    ax.bar(np.arange(6) + (k - 2.5) * w, [val(LANG[lg], c, y, col) for c in CATS], w, color=RAMP[k], edgecolor=SURF, linewidth=1.2, zorder=3)
                ax.set_xticks(range(6))
                ax.set_xticklabels(["Resumo", "Introdução", "Trab.\nrelacionados", "Metodologia", "Resultados", "Conclusão"], fontsize=9)
                ax.set_xlabel("Seção", fontsize=10)
        axes[0].set_ylabel("Probabilidade média (%)", fontsize=10)
        if tipo == "barras_secao":
            h = [Patch(facecolor=RAMP[k], label=str(y)) for k, y in enumerate(YEARS)]
            ncol = 6
        else:
            h = [Line2D([0], [0], color=COR[c], marker=MARK[c], ms=7.5, lw=2 if tipo == "linhas" else 0, label=c) if tipo == "linhas" else Patch(facecolor=COR[c], label=c) for c in CATS]
            ncol = 6
        fig.legend(handles=h, loc="lower center", ncol=ncol, frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"Pangram-Llama em {NOME[lg]}: classes 0 e 3 por seção e ano", x=0.012, ha="left", fontsize=14.5, fontweight="bold", color=INK)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        fig.savefig(OUT / f"{tipo}_{lg}.png", dpi=150, facecolor=SURF)
        plt.close(fig)
print(sorted(p.name for p in OUT.glob("*.png")))
