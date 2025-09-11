# 📝 Mineração de Texto - Reviews Americanas

Este projeto realiza **mineração de texto** no dataset de reviews da [Americanas](https://github.com/americanas-tech/b2w-reviews01/blob/main/B2W-Reviews01.csv), com o objetivo de **classificar avaliações** em três categorias:

- 😡 **Insatisfeito**  
- 😐 **Neutro**  
- 😀 **Satisfeito**  

O modelo utilizado é o **BERTimbau (BERT em português)**, da NeuralMind.

---

## 📂 Estrutura do Projeto

```bash
text-mining-reviews/
│
├── data/
│   ├── raw/               # Dados brutos (reviews.csv)
│   ├── processed/         # Dados tratados
│
├── models/                # Modelos treinados salvos
│
├── reports/               # Relatórios (gráficos, métricas)
│
├── src/                   # Código fonte
│   ├── preprocessing.py   # Funções de limpeza de texto
│   ├── dataset.py         # Carregamento e split dos dados
│   ├── train.py           # Treinamento do modelo
│   ├── evaluate.py        # Avaliação do modelo
│   ├── predict.py         # Predição de novas frases
│   └── utils.py           # Funções auxiliares
│
├── requirements.txt       # Dependências
└── README.md              # Este arquivo
