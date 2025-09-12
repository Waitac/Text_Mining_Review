"""
Avaliação do modelo treinado (BERTimbau)
- Carrega melhor checkpoint
- Gera classification_report e matriz de confusão
- Salva figura em reports/
"""

from __future__ import annotations

import os
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.dataset import load_data, prepare_data, ID2LABEL, LABEL2ID


MODEL_DIR = "models/bertimbau_reviews/best"  # salvo pelo treino.py
REPORTS_DIR = "reports"


def batched_predict(texts, tokenizer, model, batch_size: int = 64):
    """
    Faz predições em lotes para economizar memória.
    Retorna arrays de logits e preds.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            all_logits.append(out.logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    preds = logits.argmax(axis=1)
    return logits, preds


def main():
    # 1) Carregar conjunto de teste pronto do dataset.py
    df = load_data()
    _, _, test_df, _, id2label = prepare_data(df)

    class_names = [id2label[i] for i in range(len(id2label))]

    # 2) Carregar melhor modelo e tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # 3) Textos e rótulos
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    # 4) Predição em lotes
    logits, preds = batched_predict(texts, tokenizer, model, batch_size=64)

    # 5) Relatório por classe
    print(
        classification_report(
            labels, preds, target_names=class_names, digits=4
        )
    )

    # 6) Matriz de confusão
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - BERTimbau")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.show()

    # 7) (Opcional) Probabilidades médias por classe para inspeção
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    mean_conf = probs.max(axis=1).mean()
    print(f"Confiança média das predições: {mean_conf:.3f}")


if __name__ == "__main__":
    main()
