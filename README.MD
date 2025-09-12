📝 Mineração de Texto — Reviews Americanas
Projeto de análise de sentimento em português para classificar reviews em três categorias: Insatisfeito, Neutro e Satisfeito, utilizando o modelo BERTimbau (neuralmind/bert-base-portuguese-cased).

Entrada: review_title + review_text.
Rótulos: derivados de overall_rating (1–2 Negativo, 3 Neutro, 4–5 Positivo).
Saídas: melhor checkpoint, relatórios de métricas e matriz de confusão.

📂 Estrutura do projeto
bash
text-mining-reviews/
│
├── data/
│ ├── raw/ # Dados brutos (B2W-Reviews01.csv)
│ └── processed/ # (opcional) dados tratados
│
├── models/
│ └── bertimbau_reviews/ # Checkpoints; best/ após treino
│
├── reports/
│ ├── logs/ # Logs de treinamento
│ └── confusion_matrix.png # Figura gerada na avaliação
│
├── src/
│ ├── preprocessing.py # Limpeza mínima (URLs/HTML/espaços)
│ ├── dataset.py # Concatena título+texto, mapeia rótulos e faz splits
│ ├── treino.py # Fine-tuning com Trainer (métricas e checkpoints)
│ └── avaliacao.py # Classification report + matriz de confusão
│
├── requirements.txt
└── README.md

✅ Requisitos

Python 3.9+ (recomendado 3.10/3.11).

Pip e venv.

GPU CUDA opcional (acelera treino; CPU também funciona).

Instalação do ambiente
bash

criar e ativar ambiente virtual
python -m venv .venv

Windows: .venv\Scripts\activate
Linux/Mac:
source .venv/bin/activate

instalar dependências do projeto (exceto PyTorch)
pip install --upgrade pip
pip install -r requirements.txt

Instalação do PyTorch
Escolha um dos comandos conforme o ambiente. Para outras combinações (ROCm/versões específicas), use o seletor oficial de instalação do PyTorch.

CPU (sem GPU)
bash
pip install torch torchvision torchaudio

NVIDIA CUDA (exemplo CUDA 11.8; ajuste cuXXX conforme sua CUDA)
bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Verificação rápida
python
import torch
print(torch.version)
print("CUDA disponível?", torch.cuda.is_available())

📥 Dados

Baixe o CSV original (B2W-Reviews01.csv) e coloque em: data/raw/B2W-Reviews01.csv.

O pipeline seleciona as colunas necessárias, concatena review_title + review_text e cria rótulos a partir de overall_rating automaticamente.

🚀 Como executar

Preparar dados (checagem rápida opcional)
bash
python -c "from src.dataset import load_data, prepare_data; df=load_data(); tr, va, te, *_=prepare_data(df); print(tr.head()); print({k:int(v) for k,v in tr.label.value_counts().sort_index().to_dict().items()})"

Treinar o modelo (fine‑tuning BERTimbau)
bash
python -m src.treino

Usa neuralmind/bert-base-portuguese-cased

Épocas: 3 | Batch: 8 (CPU-friendly) | LR: 2e-5

Métrica do melhor modelo: F1 Macro

Salva o melhor em models/bertimbau_reviews/best e logs em reports/logs.

Avaliar no conjunto de teste
bash
python -m src.avaliacao

Carrega o melhor checkpoint

Imprime classification_report

Salva a matriz de confusão em reports/confusion_matrix.png.


Deploy

o deploy será feito com flask, no terminal rode o comando:
python app.py

🧠 Mapeamento de rótulos

1–2 → 0 (NEGATIVO)

3 → 1 (NEUTRO)

4–5 → 2 (POSITIVO)
Os mapeamentos são injetados no cabeçalho do modelo e usados em relatórios.

⚙️ Configurações úteis

Comprimento máximo: ajuste max_length=128 em src/treino.py e src/avaliacao.py conforme o tamanho médio das reviews.

Mais épocas ou early stopping: altere num_train_epochs e habilite callback de EarlyStopping em treino.py.

TensorBoard: troque report_to=["none"] por ["tensorboard"] nos argumentos de treino e rode:
bash
tensorboard --logdir reports/logs

🧪 Verificações de sanidade
Distribuição estratificada das classes por split:
bash
python -c "from src.dataset import load_data, prepare_data; df=load_data(); tr, va, te, *_=prepare_data(df); import pandas as pd; print('train:', tr.label.value_counts(normalize=True).round(3).to_dict()); print('val:', va.label.value_counts(normalize=True).round(3).to_dict()); print('test:', te.label.value_counts(normalize=True).round(3).to_dict())"

🛠️ Solução de problemas

Colunas ausentes: o CSV precisa ter review_title, review_text e overall_rating.

Treino em CPU muito lento: reduza épocas para 1–2, use batch 8 e mantenha max_length=128; considere amostrar 20–30k linhas na fase de protótipo.

Índice fantasma ("index_level_0"): o treino usa preserve_index=False ao criar o Dataset.

📄 Licença e créditos

Modelo base: BERTimbau — NeuralMind (repositório/termos aplicáveis).

Dataset: B2W-Reviews01 (Americanas).

Projeto para fins educacionais e de pesquisa.