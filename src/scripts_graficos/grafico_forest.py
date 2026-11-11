"""Forest plot da comparação pré (2020–22) × pós (2023–25): d de Cohen com IC 95% (bootstrap) por métrica, inglês e português lado a lado, agrupado por categoria.
Marcador: ● cheio = significativo (p Holm < 0,05) e robusto às 2 sensibilidades (sem 2023; ajuste por conferência); ◆ cheio = significativo mas NÃO robusto; ○ vazado = não significativo.
Texto à direita = p ajustado por Holm sobre os 26 testes. Dados: resultados/analise/testes_pre_pos.csv (src/testes_pre_pos.py). Saída: resultados/analise/forest_pre_pos.png
Uso: python3 -I src/grafico_forest.py"""
import sys
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

CATEGORIAS = {
    "Complexidade lexical": ["mtld", "hdd", "lexical_density"],
    "Previsibilidade": ["llama_bpc", "llama_surp_sd"],
    "Estilo": ["burstiness", "bigram_rep_per1k", "ai_marker_per1k", "hedge_per1k"],
    "Legibilidade": ["fre"],
    "Sintaxe": ["mdd", "connector_per1k", "sent_len_mean"],
    "Tamanho do texto (controle)": ["n_words", "n_sentences"],
}

t = pd.read_csv(ROOT / "resultados/analise/testes_pre_pos.csv")
t = t[t.familia]
NOME = {"mtld": "MTLD", "hdd": "HD-D", "lexical_density": "Densidade lexical", "llama_bpc": "Llama bits/caractere", "llama_surp_sd": "Llama desvio do surprisal",
        "burstiness": "Burstiness", "bigram_rep_per1k": "Repetição de bigramas", "ai_marker_per1k": "Marcadores de IA", "hedge_per1k": "Atenuadores",
        "fre": "Flesch Reading Ease", "mdd": "MDD", "connector_per1k": "Conectores", "sent_len_mean": "Palavras por frase"}
COR = {"en": "#2a78d6", "pt": "#eb6834"}
TIT = {"en": "Inglês", "pt": "Português"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

# posições (de cima para baixo), ordenado por |d| em inglês dentro de cada categoria; espaço entre grupos
linhas, y, grupos = [], 0.0, []
for cat, ms in CATEGORIAS.items():
    if cat.startswith("Tamanho"):
        continue
    ms = sorted(ms, key=lambda m: -abs(t[(t.language == "en") & (t.metrica == m)].d.iloc[0]))
    y0 = y
    for m in ms:
        linhas.append((m, y)); y += 1
    grupos.append((cat, y0, y - 1)); y += 0.9
ymax = y - 0.9

fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.2), sharey=True, facecolor=SURF)
for ax, lg in zip(axes, ("en", "pt")):
    ax.set_facecolor(SURF)
    ax.axvline(0, color=INK2, lw=1.1, zorder=1)
    for cat, y0, y1 in grupos:
        ax.axhspan(y0 - 0.5, y1 + 0.5, color="#f3f2ee" if grupos.index((cat, y0, y1)) % 2 == 0 else SURF, zorder=0)
    for m, yy in linhas:
        r = t[(t.language == lg) & (t.metrica == m)].iloc[0]
        sig = r.p_holm < 0.05
        ax.plot([r.d_ic_lo, r.d_ic_hi], [yy, yy], color=COR[lg], lw=2, zorder=2, solid_capstyle="butt")
        if sig and r.robusto:
            ax.plot(r.d, yy, marker="o", ms=9, mfc=COR[lg], mec=SURF, mew=0.8, zorder=3)
        elif sig:
            ax.plot(r.d, yy, marker="D", ms=8, mfc=COR[lg], mec=SURF, mew=0.8, zorder=3)
        else:
            ax.plot(r.d, yy, marker="o", ms=9, mfc=SURF, mec=COR[lg], mew=1.8, zorder=3)
        ptxt = "p < 0,001" if r.p_holm < 0.001 else "p = " + f"{r.p_holm:.3f}".replace(".", ",")
        ax.text(1.68, yy, ptxt, ha="left", va="center", fontsize=9, color=INK2)
    ax.set_xlim(-1.35, 2.35)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5])
    ax.set_ylim(ymax + 0.4, -0.9)
    ax.set_yticks([yy for _, yy in linhas])
    ax.set_yticklabels([NOME[m] for m, _ in linhas], fontsize=10)
    ax.set_title(TIT[lg], loc="left", fontsize=13, color=INK, pad=26)
    ax.set_xlabel("d de Cohen (pós − pré, em desvios-padrão) com IC 95%", fontsize=9.8)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(length=0)
    ax.text(1.68, -0.9, "p Holm", ha="left", va="center", fontsize=8.8, color=INK2, fontweight="bold")
    if lg == "en":
        for cat, y0, y1 in grupos:
            ax.text(-1.33, y0 - 0.46, cat.upper(), ha="left", va="bottom", fontsize=8, color=INK2, fontweight="bold")
    ax.annotate("← menor no pós", xy=(-0.05, -1.55), xycoords="data", ha="right", va="center", fontsize=8.8, color=INK2, annotation_clip=False)
    ax.annotate("maior no pós →", xy=(0.05, -1.55), xycoords="data", ha="left", va="center", fontsize=8.8, color=INK2, annotation_clip=False)
fig.suptitle("Mudança de 2020–22 para 2023–25: tamanho de efeito (d de Cohen) por métrica", x=0.012, ha="left", fontsize=14.5, fontweight="bold", color=INK)
h = [Line2D([0], [0], marker="o", color="none", mfc=INK2, mec=SURF, ms=9, label="significativo e robusto"),
     Line2D([0], [0], marker="D", color="none", mfc=INK2, mec=SURF, ms=8, label="significativo, não robusto"),
     Line2D([0], [0], marker="o", color="none", mfc=SURF, mec=INK2, mew=1.8, ms=9, label="não significativo")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9.8, labelcolor=INK2, bbox_to_anchor=(0.5, 0.045))
fig.text(0.012, 0.012, "Mann-Whitney por métrica e idioma (13 × 2 = 26 testes); p ajustado por Holm (FWER) sobre os 26. 'Robusto' = significativo com o mesmo sinal sem 2023 (ano de transição) e ajustando por conferência.\n"
         "d positivo = métrica maior em 2023–25. Cinza no fundo alterna as categorias linguísticas.", fontsize=8.6, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.1, 1, 0.955), w_pad=1.5)
fig.savefig(OUT / "forest_pre_pos.png", dpi=150, facecolor=SURF)
print("ok")
