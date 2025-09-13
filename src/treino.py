"""
Treino do modelo BERTimbau para classificação de reviews (CPU-friendly)
- Usa a coluna "text" criada no dataset.py
- Padding dinâmico (DataCollatorWithPadding) para reduzir custo em CPU
- Métricas (accuracy/F1-macro), melhor checkpoint e avaliação final
"""

from __future__ import annotations

import os
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from src.dataset import load_data, prepare_data


MODEL_NAME = "neuralmind/bert-base-portuguese-cased"


# --------------------------- Tokenização --------------------------------------
def tokenize_function(batch, tokenizer):
    """
    Tokeniza a coluna 'text' gerada no prepare_data.
    Mantém truncation e max_length=128. O padding será dinâmico
    via DataCollatorWithPadding (não aqui).
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )


# --------------------------- Métricas -----------------------------------------
def compute_metrics(eval_pred):
    """
    Métricas para classificação multiclasse:
    - accuracy
    - F1 macro (mais robusta a desbalanceamento)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


# --------------------------- Pipeline de treino -------------------------------
def main():
    # (Opcional) limitar threads em CPU para previsibilidade; ajuste conforme hardware
    # os.environ.setdefault("OMP_NUM_THREADS", "4")
    # os.environ.setdefault("MKL_NUM_THREADS", "4")
    # torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    # torch.set_num_interop_threads(2)

    # 1) Carregar e preparar dataset (labels e splits estratificados)
    df = load_data()
    train_df, val_df, test_df, label2id, id2label = prepare_data(df)

    # 2) Converter para datasets Hugging Face, removendo índices auxiliares
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    # 3) Tokenizer BERTimbau (cased) + padding dinâmico no collator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Remover colunas de texto para o Trainer não repassá-las
    keep = {"input_ids", "attention_mask", "label"}
    cols_to_remove = [c for c in train_ds.column_names if c not in keep]
    train_ds = train_ds.remove_columns([c for c in cols_to_remove if c != "label"])
    val_ds = val_ds.remove_columns([c for c in cols_to_remove if c != "label"])
    test_ds = test_ds.remove_columns([c for c in cols_to_remove if c != "label"])

    # 4) Modelo com cabeçalho de classificação (3 classes) e mapeamentos
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    # 5) Argumentos de treino voltados a CPU:
    # - fp16 desativado
    # - batch menor (8) para memória
    # - padding dinâmico via data_collator
    # - dataloader_num_workers maior para aproveitar múltiplos núcleos
    training_args = TrainingArguments(
        output_dir="models/bertimbau_reviews",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="reports/logs",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,  # mantém melhor e último
        seed=42,
        report_to=["none"],  # altere para ["tensorboard"] se desejar
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
    )

    # 6) Trainer com métricas e collator de padding dinâmico
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
    )

    # 7) Treinar e salvar o melhor modelo
    trainer.train()
    trainer.save_model("models/bertimbau_reviews/best")

    # 8) Avaliação final no conjunto de teste
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
