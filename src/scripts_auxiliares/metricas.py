"""MÉTRICAS OFICIAIS (v1.0) — configuração decidida em 2026-10-15 (research.md §6b/§6c).

compute(texto, lang) -> dict. `texto` é uma unidade (corpo ou resumo, ver textos.py). Mudanças em relação ao CSV pré-analisado
(que o código legado em src/legado/ reproduz):
  burstiness   : CV (σ/µ) do nº de palavras por frase, frases do spaCy, só frases com > 3 palavras alfabéticas (regex quebrava em "et al.", "Fig.").
  fre / fk     : Flesch Reading Ease com constante de Martins et al. (1996) para PT (248,835) e 206,835 para EN, sílabas via pyphen (pt_BR / en_US);
                 frases do spaCy. Flesch-Kincaid só para EN (sem adaptação PT validada) -> NaN em PT.
  LM           : Llama-3.2-3B (mesmo modelo nos dois idiomas; principal) + GPT-2 por idioma (gpt2 / pierreguillou/gpt2-small-portuguese; controle),
                 texto INTEIRO em janelas sem sobreposição de 1.023 tokens (+BOS). Reporta bits/caractere (bpc, comparável entre idiomas/tokenizadores),
                 perplexidade por token e sd do surprisal por token.
  bigram       : bigram_rep (fração, bruta, depende do comprimento) e bigram_rep_per1k (bigramas repetidos por 1.000 bigramas = 1000 × a fração média em janelas de 1.000 palavras; NaN se < 1.000 palavras).
  ttr / yules_k: mantidos (ttr depende do comprimento; yules_k = 10⁴(M2−N)/N² sobre tokens alfabéticos minúsculos); mattr100, mtld, hdd(draws=42) adicionados/mantidos.
  marcadores   : ai_marker e hedge normalizados por 1.000 palavras (+ contagem bruta); listas oficiais v2 (marcadores.py: EN com respaldo em Kobak et al. 2025; PT = correspondentes diretas), possivelmente circulares com o Pangram.
  sintaxe      : mdd = média das médias por frase; depth_mean/depth_p90 = profundidade máxima por frase (média / percentil 90); connector_per1k;
                 lexical_density = fração de palavras de conteúdo (spaCy sm).
Valores NaN quando o texto é curto demais (ex.: resumos: mattr100, bigram_rep_per1k, mtld podem faltar).
"""
import re
import warnings
from collections import Counter

import numpy as np
import pyphen
import spacy
import torch
from lexicalrichness import LexicalRichness
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts_auxiliares.marcadores import AI, HEDGE

VERSAO = "1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPACY = {"pt": "pt_core_news_sm", "en": "en_core_web_sm"}
LANG_DIR = {"portuguese": "pt", "english": "en"}
LM_NOMES = {"llama": {"pt": "meta-llama/Llama-3.2-3B", "en": "meta-llama/Llama-3.2-3B"},
            "gpt2": {"pt": "pierreguillou/gpt2-small-portuguese", "en": "gpt2"}}
JANELA = 1023
MIN_TOK = 32
ALPHA = re.compile(r"[^\W\d_]+")
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)?")
CONTEUDO = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
HY = {"pt": pyphen.Pyphen(lang="pt_BR"), "en": pyphen.Pyphen(lang="en_US")}
FRE_BASE = {"pt": 248.835, "en": 206.835}  # Martins et al. (1996) para PT; Flesch (1948) para EN
_nlp, _lm = {}, {}


def nlp(lang):
    if lang not in _nlp:
        n = spacy.load(SPACY[lang], disable=["ner", "lemmatizer"])
        n.max_length = 3_000_000
        _nlp[lang] = n
    return _nlp[lang]


def lm(nome):
    if nome not in _lm:
        tok = AutoTokenizer.from_pretrained(nome)
        kw = {"dtype": torch.float16} if "Llama" in nome else {}
        _lm[nome] = (tok, AutoModelForCausalLM.from_pretrained(nome, **kw).eval().to(DEVICE))
    return _lm[nome]


@torch.no_grad()
def lm_stats(texto, nome):
    """bpc, ppl, sd do surprisal por token sobre o texto inteiro (janelas sem sobreposição)."""
    tok, model = lm(nome)
    ids = tok(texto, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    s, nch = [], 0
    for i in range(0, len(ids), JANELA):
        c = ids[i:i + JANELA]
        if len(c) < MIN_TOK:
            continue
        c = torch.cat([torch.tensor([bos]), c])[None].to(DEVICE)
        lp = torch.log_softmax(model(c).logits[:, :-1].float(), -1).gather(-1, c[:, 1:, None])[0, :, 0]
        s.append((-lp).cpu().numpy())
        nch += len(tok.decode(c[0, 1:]))
    if not s:
        return dict(bpc=np.nan, ppl=np.nan, surp_sd=np.nan)
    s = np.concatenate(s)
    return dict(bpc=float(s.sum() / np.log(2) / nch), ppl=float(np.exp(s.mean())), surp_sd=float(s.std()))


def contar(texto, termos):
    low = texto.lower()
    return sum(len(re.findall(r"(?<![^\W\d_])" + re.escape(t) + r"(?![^\W\d_])", low)) for t in termos)


def silabas(palavra, lang):
    return max(1, len(HY[lang].positions(palavra.lower())) + 1)


def nan_se_erro(f):
    try:
        return f()
    except (ZeroDivisionError, ValueError):
        return np.nan


def compute(texto, language, com_lm=True):
    lang = LANG_DIR.get(language)
    out = {}
    doc = nlp(lang)(texto)
    palavras = [w.lower() for w in ALPHA.findall(texto)]
    n = len(palavras)
    out["n_words"] = n
    # frases (spaCy)
    toks_frase = [[t for t in s if not t.is_space] for s in doc.sents]
    pal_frase = [sum(t.is_alpha for t in s) for s in toks_frase]
    out["n_sentences"] = sum(p > 0 for p in pal_frase)
    longas = [p for p in pal_frase if p > 3]
    out["sent_len_mean"] = float(np.mean(longas)) if longas else np.nan
    out["sent_len_sd"] = float(np.std(longas)) if longas else np.nan
    out["burstiness"] = float(np.std(longas) / np.mean(longas)) if len(longas) > 1 else np.nan
    # diversidade lexical
    lex = LexicalRichness(texto)
    out["ttr"] = lex.ttr
    out["mattr100"] = nan_se_erro(lambda: lex.mattr(window_size=100))
    out["mtld"] = nan_se_erro(lambda: lex.mtld())
    out["hdd"] = nan_se_erro(lambda: lex.hdd(draws=42))
    v = Counter(Counter(palavras).values())
    out["yules_k"] = 1e4 * (sum(i * i * x for i, x in v.items()) - n) / n**2 if n else np.nan
    # repetição de bigramas
    w = [x.lower() for x in WORD.findall(texto)]
    bg = list(zip(w, w[1:]))
    out["bigram_rep"] = 1 - len(set(bg)) / len(bg) if bg else np.nan
    jan = [w[k:k + 1000] for k in range(0, len(w) - 999, 1000)]
    out["bigram_rep_per1k"] = 1000 * float(np.mean([1 - len(set(zip(x, x[1:]))) / (len(x) - 1) for x in jan])) if jan else np.nan
    # marcadores
    for nome, lista in (("ai_marker", AI[lang]), ("hedge", HEDGE[lang])):
        c = contar(texto, lista)
        out[f"{nome}_count"], out[f"{nome}_per1k"] = c, (c / n * 1000 if n else np.nan)
    # sintaxe e densidade lexical
    alpha = [t for t in doc if t.is_alpha]
    out["lexical_density"] = sum(t.pos_ in CONTEUDO for t in alpha) / len(alpha) if alpha else np.nan
    out["connector_per1k"] = (sum(1 for t in doc if t.pos_ in {"CCONJ", "SCONJ"} or t.dep_ in {"cc", "mark"}) / len(alpha) * 1000) if alpha else np.nan
    mdd, prof = [], []
    for s in toks_frase:
        if len(s) < 4:
            continue
        d = [abs(t.i - t.head.i) for t in s if t.head != t]
        if d:
            mdd.append(np.mean(d))
        mx = 0
        for t in s:
            k, c = 0, t
            while c.head != c and k < 50:
                c, k = c.head, k + 1
            mx = max(mx, k)
        prof.append(mx)
    out["mdd"] = float(np.mean(mdd)) if mdd else np.nan
    out["depth_mean"] = float(np.mean(prof)) if prof else np.nan
    out["depth_p90"] = float(np.percentile(prof, 90)) if prof else np.nan
    # legibilidade (frases do spaCy; sílabas pyphen)
    ns = out["n_sentences"]
    if n and ns:
        asl, asw = n / ns, sum(silabas(x, lang) for x in palavras) / n
        out["fre"] = FRE_BASE[lang] - 1.015 * asl - 84.6 * asw
        out["fk_grade"] = 0.39 * asl + 11.8 * asw - 15.59 if lang == "en" else np.nan
    else:
        out["fre"] = out["fk_grade"] = np.nan
    # modelos de linguagem
    if com_lm:
        for chave, nomes in LM_NOMES.items():
            for k, val in lm_stats(texto, nomes[lang]).items():
                out[f"{chave}_{k}"] = val
    return out
