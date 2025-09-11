"""
Avaliação do modelo treinado
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.dataset import load_data, prepare_data
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "models/bertimbau_reviews"

def main():
    # Carregar dados de teste
    df = load_data()
    _, _, test_df = prepare_data(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    texts = test_df["review_text"].tolist()
    labels = test_df["label"].tolist()

    # Tokenização
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # Predição
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()

    # Relatório
    print(classification_report(labels, preds, target_names=["Insatisfeito", "Neutro", "Satisfeito"]))

    # Matriz de confusão
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Insatisfeito", "Neutro", "Satisfeito"],
                yticklabels=["Insatisfeito", "Neutro", "Satisfeito"])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

if __name__ == "__main__":
    main()
