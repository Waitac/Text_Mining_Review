"""
Pré-processamento mínimo e seguro para Transformers (BERTimbau)
- Mantém acentos, caixa e pontuação
- Remove apenas ruídos óbvios (HTML/URLs/espacos)
- Oferece utilitários opcionais para análise (não usados no input do modelo)
"""

from __future__ import annotations
import re
from typing import Iterable, List, Optional

# ----------------------------- Limpeza mínima --------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_minimal(text: str) -> str:
    """
    Limpeza leve recomendada para BERT:
    - remove HTML simples
    - remove URLs
    - normaliza espaços
    NÃO: força minúsculas, remove pontuação, acentos ou stopwords.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Função principal usada no dataset:
    - aplica apenas a limpeza mínima acima.
    Adequado para tokenizers cased (ex.: BERTimbau).
    """
    return clean_minimal(text)


# ------------------------- Utilidades opcionais -------------------------------
# As funções abaixo são opcionais para análise exploratória/relatórios.
# NÃO usar o resultado delas como entrada do modelo BERT sem validação explícita.

def normalize_for_reports(
    text: str,
    lower: bool = True,
    keep_punct: bool = True,
) -> str:
    """
    Normalização leve para relatórios: útil para contagens de termos/wordclouds.
    Não recomendada como input do BERT.
    """
    t = clean_minimal(text)
    if lower:
        t = t.lower()
    if not keep_punct:
        t = re.sub(r"[^\w\sÀ-ÖØ-öø-ÿ]", " ", t, flags=re.UNICODE)
    t = _WS_RE.sub(" ", t).strip()
    return t


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenização simplista para análises (não para o modelo).
    """
    return normalize_for_reports(text).split()
