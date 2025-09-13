from __future__ import annotations
import os
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

MODEL_DIR = "models/bertimbau_reviews/best"

app = Flask(__name__)

clf = None
loaded = False

def load_pipeline():
    global clf
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # Não definir top_k aqui; definir device explicitamente
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=model, tokenizer=tok, truncation=True, device=device)

@app.before_request
def before_first_request():
    global loaded
    if not loaded:
        load_pipeline()
        loaded = True

@app.get("/")
def index():
    return render_template("index.html")

def normalize_result(pipe_output):
    """
    Normaliza diferentes formatos do pipeline para um único dicionário {label, score}.
    - Se vier lista de dicts (todas as classes), pega o de maior score.
    - Se vier lista de lista, desaninha e aplica a mesma regra.
    - Se vier um dict único, retorna direto.
    """
    out = pipe_output
    # Caso liste de listas
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]
    # Caso lista de dicts
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        best = max(out, key=lambda x: x.get("score", 0.0))
        return {"label": best.get("label"), "score": float(best.get("score", 0.0))}
    # Caso dict único
    if isinstance(out, dict):
        return {"label": out.get("label"), "score": float(out.get("score", 0.0))}
    # Fallback
    return {"label": None, "score": 0.0}

@app.post("/predict")
def predict_form():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Digite um texto para classificar.")
    # Chamar com top_k=None -> retorna lista de dicts, uma por classe
    raw = clf(text, max_length=128, top_k=None)
    result = normalize_result(raw)
    return render_template("index.html", text=text, label=result["label"], score=f"{result['score']:.3f}")

@app.post("/api/predict")
def predict_api():
    data = request.get_json(force=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Campo 'text' é obrigatório."}), 400
    raw = clf(text, max_length=128, top_k=None)
    result = normalize_result(raw)
    return jsonify({"label": result["label"], "score": float(result["score"])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
