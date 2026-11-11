"""Seleção de métricas para as ANÁLISES (as colunas continuam sendo geradas e gravadas em metricas_corpo.csv).
Fonte única de verdade: scripts de análise importam daqui. Atualizar a cada decisão (registrada em research.md §10).
Categorias do esboço: complexidade lexical, previsibilidade, estilo, legibilidade, sintaxe (+ tamanho como controle)."""

CATEGORIAS = {
    "Complexidade lexical": ["mtld", "hdd", "lexical_density"],
    "Previsibilidade": ["llama_bpc", "llama_surp_sd"],
    "Estilo": ["burstiness", "bigram_rep_per1k", "ai_marker_per1k", "hedge_per1k"],
    "Legibilidade": ["fre"],
    "Sintaxe": ["mdd", "connector_per1k", "sent_len_mean"],
    "Tamanho do texto (controle)": ["n_words", "n_sentences"],
}

# Gerada mas fora das análises, com o motivo
EXCLUIDAS = {
    "yules_k": "redundante com HD-D (ρ −0,97 EN / −0,92 PT), sinal invertido e tokenização original não reproduzida",
    "mattr100": "redundante com MTLD (ρ 0,94 EN / 0,93 PT); MTLD é a medida tradicional",
    "llama_ppl": "é o bpc em outra escala e depende do tokenizador (EN 11,2 vs PT 7,0, o oposto do bpc); bpc = perplexidade normalizada por caractere",
    "gpt2_bpc": "modelo de controle descartado como análise; GPT-2 e Llama concordam (ρ 0,80 EN / 0,76 PT); mencionar como limitação/trabalho relacionado",
    "gpt2_ppl": "dois GPT-2 diferentes por idioma (não comparáveis entre idiomas); mais fraca da categoria",
    "gpt2_surp_sd": "mais fraca e menos relacionada às demais (ρ 0,48-0,55 com o Llama)",
    "bigram_rep": "fração bruta de bigramas repetidos; depende do comprimento (ρ com n_words 0,49 EN / 0,60 PT); substituída por bigram_rep_per1k (ρ 0,88/0,86 entre as duas)",
    "ai_marker_count": "contagem bruta de marcadores; redundante com ai_marker_per1k (ρ 0,95/0,96) e dependente do comprimento",
    "hedge_count": "contagem bruta de atenuadores; redundante com hedge_per1k (ρ 0,92/0,94) e dependente do comprimento",
    "fk_grade": "existe só para EN (sem adaptação validada para PT); mesmas entradas do fre (tamanho de frase e de palavra), logo redundante com ele; fre cobre os dois idiomas",
    "depth_mean": "profundidade da árvore muito ligada ao tamanho da frase (ρ 0,80 EN / 0,86 PT com sent_len_mean); pouco acrescenta além de sent_len_mean e mdd",
    "depth_p90": "redundante com depth_mean (ρ 0,84/0,79) e com sent_len_mean (0,67/0,71)",
    "sent_len_sd": "burstiness sem normalizar pela média (ρ 0,86 EN / 0,68 PT); burstiness já está em Estilo",
    "ttr": "depende do comprimento do texto (ρ com n_words −0,68 EN / −0,74 PT); MTLD e HD-D medem a mesma ideia sem esse viés",
}

METRICAS = [m for ms in CATEGORIAS.values() for m in ms]
