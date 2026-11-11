"""Extração das unidades de texto dos JSONs estruturados (dataset_pt/, dataset_en/).

Unidades de análise (decisão de 2026-10-15, research.md §6c):
  corpo  : conteúdo (recursivo) das seções, SEM Agradecimentos, Apêndices/Anexos e seções de referências que sejam de fato listas bibliográficas.
           É `full_text` menos o(s) resumo(s) e menos o excluído.
ATENÇÃO (verificado em 2026-10-15): das 146 seções intituladas "Referências/References" só 3 contêm lista bibliográfica; as outras 143 são TEXTO DO ARTIGO
(introdução, trabalhos relacionados...) rotulado errado pelo parser e quase sempre no meio do documento — por isso o título sozinho NÃO basta para excluir referências.
  resumo : resumo no idioma do artigo (`abstract[lang]`); vazio se o artigo não o tiver (2 de 1.702).
Verificado nos 1.702 artigos: full_text == corpo bruto + resumo(s) (PT e/ou EN).
"""
import re

EXCLUIR_SECAO = re.compile(  # excluídas pelo título (Agradecimentos são curtos e sempre a última seção: 375/377)
    r"^\W*(\d+(\.\d+)*\.?\s*)?(agradecimentos?|acknowledg(e)?ments?|ap[eê]ndices?|appendix|anexos?)\b", re.I)
TITULO_REFERENCIAS = re.compile(r"^\W*(\d+(\.\d+)*\.?\s*)?(refer[eê]ncias?( bibliogr[aá]ficas)?|references|bibliography)\b", re.I)
LIM_BIB = 15  # sinais bibliográficos por 1.000 caracteres (anos, DOI/URL, "In:", Proceedings, pp., v.)


def bib_score(t):
    k = len(re.findall(r"\b(19|20)\d{2}\b", t)) + 3 * len(re.findall(r"doi|https?://|\bIn:|Proceedings|Journal|Anais|\bpp?\.\s*\d|\bv\.\s*\d", t, re.I))
    return 1000 * k / max(len(t), 1)


def _conteudo(secao):
    partes = [secao.get("sectionContent") or ""]
    for sub in secao.get("subsections") or []:
        partes.append(_conteudo(sub))
    return "\n".join(p for p in partes if p.strip())


def excluir(secao):
    t = secao.get("sectionTitle") or ""
    if EXCLUIR_SECAO.match(t):
        return True
    return bool(TITULO_REFERENCIAS.match(t)) and bib_score(_conteudo(secao)) >= LIM_BIB


def corpo(j):
    secs = [s for s in j["sections"] if not excluir(s)]
    return re.sub(r"[ \t]+", " ", "\n\n".join(_conteudo(s) for s in secs if _conteudo(s).strip())).strip()


def resumo(j, lang):
    return re.sub(r"\s+", " ", j["abstract"].get(lang) or "").strip()


def unidade(j, lang, nome):
    return {"corpo": lambda: corpo(j), "resumo": lambda: resumo(j, lang)}[nome]()
