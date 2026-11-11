"""Pangram-Llama (EditLens oficial da Pangram sobre Llama-3.2-3B) — carregamento e pontuação.

Receita de carregamento copiada de ../Pangram_test/src/detectors.py (que por sua vez segue pangramlabs/EditLens scripts/inference.py):
adapter QLoRA 4-bit sobre meta-llama/Llama-3.2-3B + cabeça LayerNorm->Linear(4). Saída: probabilidades das 4 classes
(0 humano ... 3 IA) em %, como as colunas LABEL_0..3 do CSV pré-analisado.
O token do Hugging Face (HF_TOKEN2, com acesso aprovado a meta-llama/Llama-3.2-3B e a pangram/editlens_Llama-3.2-3B, ambos "gated") é lido da variável de
ambiente HF_TOKEN2 ou do arquivo .env na raiz do pacote (ver .env.example).  Máx. 1024 tokens por texto (o modelo trunca)."""
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

BASE, ADAPTER, N = "meta-llama/Llama-3.2-3B", "pangram/editlens_Llama-3.2-3B", 4
MAX_TOK = 1024  # padrão do modelo; o script oficial usa janelas de 512 tokens (+BOS)
_m = {}


def _token():
    if os.environ.get("HF_TOKEN2"):
        return os.environ["HF_TOKEN2"]
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "HF_TOKEN2":
                return v.strip()


class _NormedLinear(torch.nn.Module):
    def __init__(self, h, n, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(h, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(h, n, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm(x))


def load():
    if "m" not in _m:
        t = _token()
        if t:
            os.environ["HF_TOKEN"] = t  # sobrescreve de propósito: os modelos "gated" só abrem com a conta da HF_TOKEN2
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(BASE)
        if tok.pad_token is None:
            tok.pad_token, tok.padding_side = tok.eos_token, "left"
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE, num_labels=N, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
        base.config.pad_token_id = tok.pad_token_id
        base.score = _NormedLinear(base.config.hidden_size, N, device=next(base.parameters()).device, dtype=torch.bfloat16)
        _m["m"] = (tok, PeftModel.from_pretrained(base, ADAPTER).eval())
    return _m["m"]


@torch.no_grad()
def probs(texts, batch=4, max_length=MAX_TOK):
    """Lista de textos -> array (len, 4) com probabilidades em %."""
    tok, model = load()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
        out += (torch.softmax(model(**enc).logits.float(), -1) * 100).cpu().tolist()
    return out
