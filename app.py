from __future__ import annotations
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, render_template, jsonify, g
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from collections import Counter
import re

MODEL_DIR = "models/bertimbau_reviews/best"
CSV_PATH = "/home/wmoura/Projetos Waita/Text Mining/data/raw/B2W-Reviews01.csv"
DATABASE = "reviews.db"

app = Flask(__name__)

clf = None
loaded = False

# ===== Database Setup =====
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                product_brand TEXT NOT NULL,
                review_text TEXT NOT NULL,
                sentiment_label TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()

# ===== Load Model =====
def load_pipeline():
    global clf
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=model, tokenizer=tok, truncation=True, device=device)

@app.before_request
def before_first_request():
    global loaded
    if not loaded:
        load_pipeline()
        loaded = True

# ===== Load CSV Data =====
def load_brands():
    try:
        df = pd.read_csv(CSV_PATH, usecols=['product_brand'])
        df = df.dropna(subset=['product_brand'])
        brands = sorted(df['product_brand'].unique().tolist())
        return brands
    except Exception as e:
        print(f"Erro ao carregar marcas do CSV: {e}")
        return []

def load_products_by_brand(brand):
    try:
        df = pd.read_csv(CSV_PATH, usecols=['product_name', 'product_brand'])
        df = df.dropna(subset=['product_name', 'product_brand'])
        df_filtered = df[df['product_brand'] == brand]
        products = sorted(df_filtered['product_name'].unique().tolist())
        return products
    except Exception as e:
        print(f"Erro ao carregar produtos do CSV: {e}")
        return []

# ===== Helper Functions =====
def normalize_result(pipe_output):
    out = pipe_output
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        best = max(out, key=lambda x: x.get("score", 0.0))
        return {"label": best.get("label"), "score": float(best.get("score", 0.0))}
    if isinstance(out, dict):
        return {"label": out.get("label"), "score": float(out.get("score", 0.0))}
    return {"label": None, "score": 0.0}

def extract_keywords(text):
    """Extrai palavras relevantes (removendo stopwords básicas)"""
    stopwords = {'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 
                 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas',
                 'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já',
                 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois',
                 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você',
                 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas','tem'}
    
    words = re.findall(r'\b[a-záàâãéèêíïóôõöúçñ]{3,}\b', text.lower())
    return [w for w in words if w not in stopwords]

# ===== Routes =====
@app.get("/")
def index():
    brands = load_brands()
    return render_template("index.html", brands=brands)

@app.get("/api/products/<brand>")
def get_products(brand):
    products = load_products_by_brand(brand)
    return jsonify({"products": products})

@app.post("/predict")
def predict_form():
    product = request.form.get("product", "").strip()
    brand = request.form.get("brand", "").strip()
    text = request.form.get("text", "").strip()
    
    brands = load_brands()
    
    if not product or not brand or not text:
        return render_template("index.html", brands=brands, 
                             error="Preencha todos os campos.")
    
    # Predição
    raw = clf(text, max_length=128, top_k=None)
    result = normalize_result(raw)
    
    # Salvar no banco
    db = get_db()
    db.execute('''
        INSERT INTO reviews (product_name, product_brand, review_text, sentiment_label, sentiment_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (product, brand, text, result["label"], result["score"]))
    db.commit()
    
    return render_template("index.html", brands=brands,
                         selected_brand=brand, selected_product=product, text=text, 
                         label=result["label"], score=f"{result['score']:.3f}",
                         success="Review salva com sucesso!")

@app.get("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.get("/api/dashboard-data")
def dashboard_data():
    db = get_db()
    
    # Total de reviews por sentimento
    sentiment_counts = db.execute('''
        SELECT sentiment_label, COUNT(*) as count
        FROM reviews
        GROUP BY sentiment_label
    ''').fetchall()
    
    # Sentimento por marca (top 10 marcas com mais reviews)
    brand_sentiment = db.execute('''
        SELECT product_brand, sentiment_label, COUNT(*) as count
        FROM reviews
        GROUP BY product_brand, sentiment_label
        ORDER BY product_brand
    ''').fetchall()
    
    # Processar para agrupar por marca
    brand_data = {}
    for row in brand_sentiment:
        brand = row['product_brand']
        if brand not in brand_data:
            brand_data[brand] = {'NEGATIVO': 0, 'NEUTRO': 0, 'POSITIVO': 0, 'total': 0}
        brand_data[brand][row['sentiment_label']] = row['count']
        brand_data[brand]['total'] += row['count']
    
    # Top 10 marcas por total de reviews
    top_brands = sorted(brand_data.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
    
    # Distribuição de confiança por sentimento
    confidence_dist = db.execute('''
        SELECT sentiment_label, sentiment_score
        FROM reviews
    ''').fetchall()
    
    # Criar histograma de confiança
    confidence_buckets = {}
    for row in confidence_dist:
        label = row['sentiment_label']
        score = row['sentiment_score']
        bucket = round(score * 10) / 10  # Arredondar para 0.1
        
        if label not in confidence_buckets:
            confidence_buckets[label] = {}
        if bucket not in confidence_buckets[label]:
            confidence_buckets[label][bucket] = 0
        confidence_buckets[label][bucket] += 1
    
    # Palavras-chave mais frequentes por sentimento
    all_reviews = db.execute('SELECT review_text, sentiment_label FROM reviews').fetchall()
    keywords_by_sentiment = {'NEGATIVO': [], 'NEUTRO': [], 'POSITIVO': []}
    
    for row in all_reviews:
        keywords = extract_keywords(row['review_text'])
        keywords_by_sentiment[row['sentiment_label']].extend(keywords)
    
    # Top 15 palavras por sentimento
    top_keywords = {}
    for sentiment, words in keywords_by_sentiment.items():
        counter = Counter(words)
        top_keywords[sentiment] = counter.most_common(15)
    
    # Reviews ao longo do tempo (últimos 30 dias)
    timeline = db.execute('''
        SELECT DATE(created_at) as date, sentiment_label, COUNT(*) as count
        FROM reviews
        WHERE created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at), sentiment_label
        ORDER BY date
    ''').fetchall()
    
    # Processar timeline por sentimento
    timeline_data = {}
    for row in timeline:
        date = row['date']
        if date not in timeline_data:
            timeline_data[date] = {'NEGATIVO': 0, 'NEUTRO': 0, 'POSITIVO': 0}
        timeline_data[date][row['sentiment_label']] = row['count']
    
    return jsonify({
        'sentiment_counts': [dict(row) for row in sentiment_counts],
        'brand_sentiment': [(brand, data) for brand, data in top_brands],
        'confidence_buckets': confidence_buckets,
        'top_keywords': top_keywords,
        'timeline': timeline_data
    })

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
