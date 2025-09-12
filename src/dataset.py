"""
Módulo para carregar e preparar o dataset de reviews (BERTimbau)
- Usa review_title + review_text como entrada textual
- Gera rótulos triclasse a partir de overall_rating
- Realiza splits estratificados e reprodutíveis
"""

from __future__ import annotations
import re
from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------- Utilidades básicas -----------------------------
def _clean_minimal(text: str) -> str:
    """
    Limpeza mínima segura para Transformers:
    - remove tags HTML simples
    - remove URLs
    - normaliza espaços
    NÃO altera acentuação ou pontuação.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = re.sub(r"<[^>]+>", " ", text)              # HTML simples
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"\s+", " ", text).strip()          # espaços
    return text


def _concat_title_text(title: str, body: str) -> str:
    """
    Concatena título e corpo preservando contexto.
    - Se ambos existem: 'title — body'
    - Se um faltar: retorna o outro
    """
    t = _clean_minimal(title)
    b = _clean_minimal(body)
    if t and b:
        return f"{t} — {b}"
    return t or b


def _map_rating_to_label(r: float) -> int:
    """
    Mapeia estrelas para classes:
    1-2 -> 0 (NEGATIVO), 3 -> 1 (NEUTRO), 4-5 -> 2 (POSITIVO)
    Lida com valores não numéricos retornando None (será filtrado).
    """
    try:
        r = float(r)
    except Exception:
        return None
    if r <= 2:
        return 0
    if r == 3:
        return 1
    if r >= 4:
        return 2
    return None


LABEL2ID: Dict[str, int] = {"NEGATIVO": 0, "NEUTRO": 1, "POSITIVO": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


# --------------------------------- I/O ---------------------------------------
def load_data(path: str = "data/raw/B2W-Reviews01.csv") -> pd.DataFrame:
    """
    Carrega o CSV bruto.
    Espera encontrar as colunas:
      - review_title
      - review_text
      - overall_rating (1 a 5)
    """
    df = pd.read_csv(path)
    return df


# ------------------------------ Preparação -----------------------------------
def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Prepara dados para fine-tuning com BERTimbau:
    1) Valida colunas necessárias
    2) Concatena review_title + review_text em 'text'
    3) Gera coluna 'label' (0=NEG, 1=NEU, 2=POS) a partir de overall_rating
    4) Remove nulos/duplicados e linhas sem rótulo
    5) Realiza splits estratificados: train/val/test
    Retorna: train, val, test, LABEL2ID, ID2LABEL
    """

    required_cols = {"review_title", "review_text", "overall_rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no DataFrame: {sorted(missing)}. "
            "Certifique-se de que o CSV contenha review_title, review_text e overall_rating."
        )

    # 2) Texto de entrada: título + texto
    df = df.copy()
    df["text"] = [
        _concat_title_text(t, x) for t, x in zip(df["review_title"], df["review_text"])
    ]

    # 3) Rótulos a partir das estrelas
    df["label"] = df["overall_rating"].apply(_map_rating_to_label)

    # 4) Higiene: descartar linhas inválidas
    df = df.dropna(subset=["text", "label"]).copy()
    df = df[df["text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    # Checagem de distribuição antes do split
    label_counts = df["label"].value_counts().sort_index()
    if label_counts.min() == 0:
        raise ValueError(
            "Alguma classe ficou vazia após o pré-processamento. "
            "Revise o mapeamento de rótulos ou os filtros."
        )

    # 5) Split estratificado: primeiro separa teste, depois validação a partir do treino
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
        shuffle=True,
    )
    # fração de validação relativa ao conjunto de treino resultante
    val_relative = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_relative,
        random_state=random_state,
        stratify=train_df["label"],
        shuffle=True,
    )

    # Seleciona apenas colunas necessárias para o modelo
    cols_keep = ["text", "label"]
    train_df = train_df[cols_keep].reset_index(drop=True)
    val_df = val_df[cols_keep].reset_index(drop=True)
    test_df = test_df[cols_keep].reset_index(drop=True)

    return train_df, val_df, test_df, LABEL2ID, ID2LABEL
