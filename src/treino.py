"""
Treino do modelo BERTimbau para classificação de reviews
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from src.dataset import load_data, prepare_data

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

def tokenize_data(examples, tokenizer):
    return tokenizer(examples["review_text"], truncation=True, padding="max_length", max_length=128)

def main():
    # Carregar e preparar dataset
    df = load_data()
    train_df, val_df, test_df = prepare_data(df)

    # Converter para formato HuggingFace Dataset
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = train_ds.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_data(x, tokenizer), batched=True)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # Configurações de treino
    training_args = TrainingArguments(
        output_dir="models/",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="reports/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("models/bertimbau_reviews")

if __name__ == "__main__":
    main()
