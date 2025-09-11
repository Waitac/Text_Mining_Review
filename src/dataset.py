"""
Módulo para carregar e preparar o dataset de reviews
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_text

def load_data(path="data/raw/B2W-Reviews01.csv"):
    """Carrega dataset e retorna DataFrame"""
    df = pd.read_csv(path)
    return df

def prepare_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Limpa texto e divide em treino, validação e teste"""
    df["review_text"] = df["review_text"].astype(str).apply(preprocess_text)

    # Exemplo: converter nota em classes (ajuste conforme dataset)
    # nota >= 4 -> satisfeito, nota == 3 -> neutro, nota <= 2 -> insatisfeito
    df["label"] = df["rating"].apply(
        lambda x: 0 if x <= 2 else (1 if x == 3 else 2)
    )

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train, test_size=val_size, random_state=random_state)

    return train, val, test
