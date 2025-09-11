"""
Módulo de pré-processamento de texto em português
- Limpeza de caracteres
- Stopwords
- Lematização (spaCy)
"""

import re
import spacy
import nltk

# Baixar stopwords PT-BR do NLTK (apenas na primeira vez)
nltk.download("stopwords")
stopwords = set(nltk.corpus.stopwords.words("portuguese"))

# Carregar modelo spaCy para PT-BR
nlp = spacy.load("pt_core_news_sm")

def clean_text(text):
    """Remove caracteres especiais, links e coloca em minúsculas"""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)   # remove links
    text = re.sub(r"[^a-zà-ú ]", "", text) # mantém apenas letras
    return text.strip()

def preprocess_text(text):
    """Aplica limpeza, remove stopwords e lematiza"""
    text = clean_text(text)
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if token.lemma_ not in stopwords and not token.is_punct
    ]
    return " ".join(tokens)
