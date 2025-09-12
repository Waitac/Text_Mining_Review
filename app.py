from __future__ import annotations
import os
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Caminho do melhor checkpoint salvo pelo treino.py
MODEL_DIR = "models/bertimbau_reviews/best"

# Cria app Flask
app = Flask(__name__)

# Carrega pipeline uma única vez no startup
clf = None

def load_pipeline():
    global clf
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # text-classification retorna [{"label": "POSITIVO", "score": 0.95}, ...]
    clf = pipeline("text-classification", model=model, tokenizer=tok, top_k=None, truncation=True)

@app.before_first_request
def _startup():
    load_pipeline()

# Página com formulário
@app.get("/")
def index():
    return render_template("index.html")

# Submissão do formulário (HTML)
@app.post("/predict")
def predict_form():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Digite um texto para classificar.")
    result = clf(text, max_length=128)[0]
    # result: {'label': 'NEGATIVO'|'NEUTRO'|'POSITIVO', 'score': float}
    return render_template("index.html", text=text, label=result["label"], score=f"{result['score']:.3f}")

# Endpoint JSON (útil para integrações)
@app.post("/api/predict")
def predict_api():
    data = request.get_json(force=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Campo 'text' é obrigatório."}), 400
    result = clf(text, max_length=128)[0]
    return jsonify({"label": result["label"], "score": float(result["score"])})

if __name__ == "__main__":
    # Em produção, usar gunicorn/uwsgi; debug=True apenas localmente
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
