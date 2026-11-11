"""Listas OFICIAIS de marcadores lexicais associados a LLMs (AI) e de atenuadores epistêmicos (HEDGE) — v2, decidido em 2026-10-15.

Origem: listas originais dos coautores (referencias/ai_markers_and_hedges_original.py; reproduzem 100% das contagens do CSV pré-analisado), com estes critérios:
  MARCADORES EN: só famílias de palavras com RESPALDO NA LITERATURA = presentes (formas exatas ou flexões) na lista de excesso de vocabulário de
    Kobak et al. (2025, Science Advances 11(27):eadt3813; referencias/kobak_excess_words.csv). Ficam 18 famílias; saem vital, paramount, synergy, furthermore, moreover,
    robust, holistic (ausentes em Kobak) e as frases/hifenizadas (it is worth noting..., shed light, in the context of, cutting-edge, state-of-the-art), não comparáveis com uma lista de palavras.
    As flexões são as da lista original mais as formas de Kobak com a mesma raiz.
  MARCADORES PT: correspondentes diretas (com flexões de gênero/número/pessoa) das famílias EN adotadas que já existiam na lista original em português (aprofundar, alavancar,
    abrangente, multifacetado, notável) MAIS os candidatos dos exemplos do esboço (fundamental, crucial, fomentar, âmbito, panorama, sutil, ressaltar, inovador, sinergia, de vanguarda),
    incluídos por decisão do usuário em 2026-10-15; sinergia e "de vanguarda" não têm respaldo encontrado na literatura (ver RESPALDO). Atenuadores PT: os exemplos do esboço
    (poderia, sugere, possivelmente, talvez) já estavam contemplados.
  ATENUADORES EN: a lista original completa (14 itens). ATENUADORES PT: as correspondentes diretas na lista original em português, com flexões.
Contagem: palavra/expressão inteira, sem diferenciar maiúsculas (metricas.contar). Para evitar contagem dupla, expressões contidas em outras da mesma lista foram removidas ("parece que" ⊂ "parece").
O que ficou de fora e PODE ser importante está em PENDENTES_* (NÃO usado nas métricas; decisão futura).
"""

# ---------------------------------------------------------------- marcadores EN
AI_EN = {  # família -> formas
    "delve": ["delve", "delves", "delving", "delved"],
    "leverage": ["leverage", "leverages", "leveraging"],
    "pivotal": ["pivotal"],
    "crucial": ["crucial"],
    "underscore": ["underscore", "underscores", "underscored", "underscoring"],
    "foster": ["foster", "fosters", "fostering"],
    "realm": ["realm", "realms"],
    "landscape": ["landscape"],
    "commendable": ["commendable"],
    "noteworthy": ["noteworthy"],
    "additionally": ["additionally"],
    "consequently": ["consequently"],
    "comprehensive": ["comprehensive"],
    "thorough": ["thorough"],
    "showcase": ["showcase", "showcases", "showcased", "showcasing"],
    "multifaceted": ["multifaceted"],
    "nuanced": ["nuanced", "nuances"],
    "groundbreaking": ["groundbreaking"],
}
# ---------------------------------------------------------------- marcadores PT (correspondência direta com flexões)
AI_PT = {  # chave = família EN adotada correspondente (ou "(esboço) ..." quando só vem do esboço) -> formas em português
    # correspondentes diretas que já estavam na lista original em português
    "delve": ["aprofundar", "aprofunda", "aprofundam", "aprofundando", "aprofundou", "aprofundado", "aprofundada", "aprofundados", "aprofundadas"],
    "leverage": ["alavancar", "alavanca", "alavancam", "alavancando", "alavancou", "alavancado", "alavancada", "alavancados", "alavancadas", "alavancagem"],
    "comprehensive": ["abrangente", "abrangentes"],
    "multifaceted": ["multifacetado", "multifacetada", "multifacetados", "multifacetadas"],
    "noteworthy": ["notável", "notáveis"],
    # candidatos que aparecem na lista de exemplos do esboço (incluídos em 2026-10-15)
    "pivotal": ["fundamental", "fundamentais"],
    "crucial": ["crucial", "cruciais"],
    "foster": ["fomentar", "fomenta", "fomentam", "fomentando", "fomentou", "fomentado", "fomentada", "fomentados", "fomentadas"],
    "realm": ["âmbito", "âmbitos"],
    "landscape": ["panorama", "panoramas"],
    "nuanced": ["sutil", "sutis"],
    "underscore": ["ressaltar", "ressalta", "ressaltam", "ressaltando", "ressaltou", "ressaltado", "ressaltada", "ressaltados", "ressaltadas"],
    "(esboço) innovative": ["inovador", "inovadora", "inovadores", "inovadoras"],
    "(esboço) synergy": ["sinergia", "sinergias"],
    "(esboço) cutting-edge": ["de vanguarda"],
}
RESPALDO = {  # respaldo encontrado na literatura (verificado por leitura das listas, 2026-10-15); "—" = não encontrado nas fontes lidas
    "Kobak et al. (2025) [lista de 900 palavras]": "delve, leverage(s/ing), pivotal, crucial, underscore, foster(s/ing), realm, landscape, commendable, noteworthy, additionally, consequently, comprehensive, thorough, showcase, multifaceted, nuanced, groundbreaking, innovative, transformative, pioneering, notable",
    "Juzek & Ward (COLING 2025) [21 palavras focais]": "delve, underscore, showcase(s/ing), realm, groundbreaking",
    "Geng & Trotta (2025), Tabela 1 [citando Liang et al. 2024a/b e Geng & Trotta 2024]": "innovative, notable, commendable, meticulous, pivotal, comprehensive, crucial, significant",
    "—": "robust, holistic, synergy, paramount, vital, furthermore, moreover, cutting-edge, state-of-the-art, 'in the context of', 'it is worth noting'; PT: robusto, holístico, sinergia, paradigmático, de ponta/de vanguarda, primordial, vale ressaltar/destacar",
}
# ---------------------------------------------------------------- atenuadores
HEDGE_EN = {  # item original -> formas em inglês
    "may": ["may"], "might": ["might"], "could": ["could"], "potentially": ["potentially"], "possibly": ["possibly"], "perhaps": ["perhaps"],
    "it seems": ["it seems"], "appear to": ["appear to"], "tend to": ["tend to"], "generally": ["generally"], "typically": ["typically"],
    "in general": ["in general"], "it is suggested": ["it is suggested"], "it is proposed": ["it is proposed"],
}
HEDGE_PT = {  # item EN -> correspondentes na lista original em português (com flexões); [] = sem correspondente na lista original
    "may": ["poderia", "poderiam"], "might": ["poderia", "poderiam"], "could": ["poderia", "poderiam"],  # compartilham as mesmas formas; contadas uma vez
    "potentially": [], "possibly": ["possivelmente"], "perhaps": ["talvez"],
    "it seems": ["parece", "parecem"], "appear to": ["parece", "parecem"],                              # idem
    "tend to": ["tende a", "tendem a"], "generally": ["geralmente"], "typically": ["tipicamente"], "in general": ["em geral"],
    "it is suggested": ["sugere", "sugerem"], "it is proposed": [],
}

# ---------------------------------------------------------------- listas planas usadas por metricas.py (sem repetição)
def _plano(d):
    out = []
    for formas in d.values():
        for f in formas:
            if f not in out:
                out.append(f)
    return out


AI = {"en": _plano(AI_EN), "pt": _plano(AI_PT)}
HEDGE = {"en": _plano(HEDGE_EN), "pt": _plano(HEDGE_PT)}

# ---------------------------------------------------------------- PENDENTES (não usados) — decidir depois
PENDENTES_AI_EN_SEM_RESPALDO_KOBAK = ["vital", "paramount", "synergy/synergies/synergistic", "furthermore", "moreover", "robust", "holistic"]
PENDENTES_AI_EN_FRASES = ["it is worth noting", "it is important to note", "it should be noted", "shed light / sheds light", "in the context of", "cutting-edge", "state-of-the-art"]
PENDENTES_AI_PT_CANDIDATAS = {  # famílias EN adotadas ainda SEM equivalente em português (o esboço não os traz)
    "commendable": "louvável", "additionally": "adicionalmente", "consequently": "consequentemente", "thorough": "minucioso, rigoroso",
    "showcase": "evidenciar, apresentar", "groundbreaking": "revolucionário, pioneiro",
}
PENDENTES_AI_PT_SO_NA_LISTA_ORIGINAL = ["transformador/a/es (respaldo: transformative em Kobak)", "pioneiro/a/os (respaldo: pioneering em Kobak)", "robusto/a/os/as (sem respaldo encontrado)",
                                       "holístico/a/os/as (sem respaldo)", "sinérgico/a (sem respaldo)", "paradigmático/a (sem respaldo)", "de ponta (sem respaldo)", "primordial/is (sem respaldo)",
                                       "vale ressaltar/destacar, cabe ressaltar/destacar, é importante notar/destacar (frases; sem respaldo)", "no contexto de (sem respaldo)"]
PENDENTES_HEDGE_EN_FLEXOES = ["appears to", "appeared to", "seems to", "tends to", "tended to", "might/could/may + have", "it is suggested that (variantes: it is suggested, suggests, suggest, indicates)"]
PENDENTES_HEDGE_PT_AUSENTES_NA_LISTA_ORIGINAL = ["pode/podem/poderá (equivalentes de 'may', o atenuador mais frequente em inglês)", "potencialmente", "propõe-se", "sugere-se (impessoal)", "aparenta(m)"]
PENDENTES_HEDGE_PT_SO_NA_LISTA_ORIGINAL = ["provavelmente", "aparentemente", "em alguns casos", "em certos casos", "pode ser que", "é possível que", "parece que (duplicaria 'parece')"]
