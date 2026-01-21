from flask import Flask, render_template, request, jsonify
import torch
import time
from transformers import AutoTokenizer
import torch.nn as nn
import sqlite3
import json
from datetime import datetime

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "answerdotai/ModernBERT-base"
LABELS = ["Negatif", "Netral", "Positif"]
DB_NAME = "sentiment_history.db"

# =========================
# DATABASE SETUP
# =========================
def init_db():
    """Membuat tabel history jika belum ada"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text_input TEXT,
            model_name TEXT,
            best_label TEXT,
            best_value REAL,
            execution_time REAL,
            all_results TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Jalankan inisialisasi DB saat aplikasi start
init_db()

# =========================
# MODEL CLASS (WAJIB SAMA)
# =========================
class SentimentModel(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.out(pooled)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================
# LOAD MODEL
# =========================
# Pastikan folder 'model/' dan file .pth benar-benar ada
model_paths = {
    "baseline": "model/modernbert_baseline_best.pth",
    "sfs": "model/modernbert_sfs_best.pth"
}

models = {}

# Try-Except block agar app tidak crash jika model belum ada (untuk testing)
try:
    for key, path in model_paths.items():
        print(f"Loading {key} from {path}...")
        model = SentimentModel()
        model.load_state_dict(
            torch.load(path, map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval()
        models[key] = model
    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Pastikan file model .pth tersedia di folder 'model/'.")

# =========================
# HELPER FUNCTIONS
# =========================
def save_to_history(text, model_used, best_label, best_val, exec_time, all_res):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Simpan all_results sebagai JSON string
    json_results = json.dumps(all_res)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO history (timestamp, text_input, model_name, best_label, best_value, execution_time, all_results)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, text, model_used, best_label, best_val, exec_time, json_results))
    
    conn.commit()
    conn.close()

def get_all_history():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Agar bisa akses kolom by name
    c = conn.cursor()
    c.execute("SELECT id, timestamp, text_input, model_name, best_label, best_value FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text_input = ""
    selected_model = "baseline"

    if request.method == "POST":
        text_input = request.form.get("text", "")
        selected_model = request.form.get("model", "baseline")

        if text_input and selected_model in models:
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            start = time.time()
            with torch.no_grad():
                logits = models[selected_model](
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )
                probs = torch.softmax(logits, dim=1)[0]
            end = time.time()

            percentages = {
                LABELS[i]: round(probs[i].item() * 100, 2)
                for i in range(3)
            }

            best_label = max(percentages, key=percentages.get)
            exec_time = round(end - start, 4)

            result = {
                "model": selected_model.upper(),
                "best_label": best_label,
                "best_value": percentages[best_label],
                "all_results": percentages,
                "time": exec_time
            }

            # SIMPAN KE DATABASE
            save_to_history(
                text_input, 
                selected_model, 
                best_label, 
                result["best_value"], 
                exec_time, 
                percentages
            )

    # Ambil riwayat untuk ditampilkan di tabel
    history_data = get_all_history()

    return render_template(
        "index.html",
        result=result,
        text_input=text_input,
        selected_model=selected_model,
        history=history_data
    )

# ROUTE BARU: API untuk mengambil detail riwayat (AJAX)
@app.route("/history/<int:history_id>", methods=["GET"])
def get_history_detail(history_id):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE id = ?", (history_id,))
    row = c.fetchone()
    conn.close()

    if row:
        # Konversi Row object ke dictionary agar bisa di-JSON
        data = dict(row)
        # Parse kolom JSON string kembali ke object asli
        data['all_results'] = json.loads(data['all_results'])
        return jsonify(data)
    else:
        return jsonify({"error": "Data not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)