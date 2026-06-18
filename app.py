import csv
import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

SPACY_AVAILABLE = False
nlp_ner = None

try:
    import spacy

    try:
        nlp_ner = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

BASE_MODEL_NAME = "distilroberta-base"
MODEL_DIR = Path("models/trafficking-roberta")
MODEL_NAME = str(MODEL_DIR)
MODEL_LOADED = False
tokenizer = None
model = None
city_counts = Counter()

CITY_COORDS = {
    "Mumbai": [19.0760, 72.8777],
    "Delhi": [28.6139, 77.2090],
    "Kolkata": [22.5726, 88.3639],
    "Chennai": [13.0827, 80.2707],
    "Bangalore": [12.9716, 77.5946],
    "Bengaluru": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Ahmedabad": [23.0225, 72.5714],
    "Pune": [18.5204, 73.8567],
    "Surat": [21.1702, 72.8311],
    "Patna": [25.5941, 85.1376],
    "Guwahati": [26.1445, 91.7362],
    "Siliguri": [26.7271, 88.3953],
    "Varanasi": [25.3176, 82.9739],
    "Lucknow": [26.8467, 80.9462],
    "Agra": [27.1767, 78.0081],
    "Jaipur": [26.9124, 75.7873],
    "Bhopal": [23.2599, 77.4126],
    "Indore": [22.7196, 75.8577],
    "Nagpur": [21.1458, 79.0882],
    "Raipur": [21.2514, 81.6296],
    "Bhubaneswar": [20.2961, 85.8245],
    "Visakhapatnam": [17.6868, 83.2185],
    "Kochi": [9.9312, 76.2673],
    "Thiruvananthapuram": [8.5241, 76.9366],
    "Coimbatore": [11.0168, 76.9558],
    "Ranchi": [23.3441, 85.3096],
    "Dhanbad": [23.7957, 86.4304],
    "Gorakhpur": [26.7606, 83.3732],
    "Kanpur": [26.4499, 80.3319],
    "Amritsar": [31.6340, 74.8723],
    "Jammu": [32.7266, 74.8570],
    "Imphal": [24.8170, 93.9368],
    "Agartala": [23.8315, 91.2868],
    "Dibrugarh": [27.4728, 94.9120],
    "Silchar": [24.8333, 92.7789],
}

RISK_KEYWORDS = {
    "control": [
        "agent manage",
        "handler",
        "supervised",
        "someone else books",
        "manager handles",
        "party arrange",
    ],
    "movement": [
        "new to this city",
        "just arrived",
        "leaving tomorrow",
        "passing through",
        "transit",
        "only here for",
        "touring",
        "shifting city",
    ],
    "age": [
        "18 saal",
        "19 saal",
        "fresh girl",
        "first time",
        "young and new",
        "village girl",
        "innocent",
    ],
    "isolation": [
        "no family here",
        "runaway",
        "nobody knows",
        "limited hindi",
        "no papers",
        "document nahi",
        "apna ghar nahi",
    ],
    "transaction": [
        "advance lena",
        "paise pehle",
        "cash only",
        "rate fix karo",
        "service charge",
        "negotiable",
        "room included",
    ],
    "secrecy": [
        "police mat lao",
        "no police",
        "seedha aao",
        "private",
        "kisi ko mat batana",
        "discreet",
        "secret",
        "no questions",
        "back door entry",
    ],
}

CATEGORY_WEIGHTS = {
    "control": 0.2,
    "movement": 0.16,
    "age": 0.18,
    "isolation": 0.18,
    "transaction": 0.12,
    "secrecy": 0.16,
}

NEGATIVE_CONTEXT = [
    "awareness",
    "workshop",
    "conference",
    "training",
    "helpline",
    "prevention",
    "policy",
    "research",
    "report",
    "ngo",
]

DEMO_TEXT = (
    "New girl just arrived in Delhi from village. Handler will manage booking, "
    "cash only, no police, room included. Leaving tomorrow for Jaipur."
)

DEMO_MAP_RECORDS = [
    "New worker just arrived in Delhi from village. Handler manages phone, cash only, no police, room included.",
    "Fresh girl in Mumbai, no family here, manager handles booking, private entry, leaving tomorrow.",
    "Young and new person in Kolkata, cash only, no questions, handler controls all travel.",
    "Village girl brought to Chennai, no papers, supervised by agent, room included, private booking.",
    "First time in Bangalore, handler manages all calls, cash only, back door entry.",
    "New to this city in Hyderabad, no family here, no police, service charge fixed by manager.",
    "Just arrived in Ahmedabad from village, supervised travel, cash only, secret booking.",
    "Fresh girl in Pune, manager handles phone, private room included, no questions.",
    "New worker in Surat, handler controls transport, no papers, cash only.",
    "Young and new in Patna, no family here, leaving tomorrow, no police.",
    "Village girl in Guwahati, agent manage, room included, private entry.",
    "Just arrived in Siliguri, handler controls booking, limited Hindi, cash only.",
    "First time in Varanasi, no papers, no questions, manager handles meeting.",
    "New to Lucknow, handler manages phone, cash only, discreet arrival.",
    "Fresh girl in Agra, no family here, room included, back door entry.",
    "Young and new in Jaipur, supervised by manager, no police, cash only.",
    "Just arrived in Bhopal, no papers, private booking, handler controls travel.",
    "Village girl in Indore, manager handles all calls, no questions, leaving tomorrow.",
    "New to Nagpur, handler manages booking, cash only, secret entry.",
    "First time in Raipur, no family here, no police, room included.",
]

app = Flask(__name__)


def load_model():
    global MODEL_NAME, MODEL_LOADED, tokenizer, model

    try:
        if MODEL_DIR.exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            MODEL_NAME = str(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            model.eval()
            MODEL_LOADED = True
        else:
            MODEL_NAME = f"{BASE_MODEL_NAME} fine-tune pending"
            MODEL_LOADED = False
    except Exception as exc:
        print(f"Model load error: {exc}")
        MODEL_LOADED = False


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", "[PHONE]", text)
    text = re.sub(r"\$\d+|\d+\s*(?:roses?|donation|tribute)", "[PRICE]", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_phones(text):
    return re.findall(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", text or "")


def extract_cities(text):
    found = []
    lower = (text or "").lower()
    for city in CITY_COORDS:
        if city.lower() in lower:
            found.append(city)
    if SPACY_AVAILABLE and nlp_ner:
        try:
            doc = nlp_ner((text or "")[:1200])
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"] and ent.text not in found:
                    found.append(ent.text)
        except Exception:
            pass
    return sorted(set(found))


def extract_persons(text):
    if not SPACY_AVAILABLE or not nlp_ner:
        return []
    try:
        doc = nlp_ner((text or "")[:1200])
        return sorted(set(ent.text for ent in doc.ents if ent.label_ == "PERSON"))
    except Exception:
        return []


def detect_risk_flags(text):
    lower = (text or "").lower()
    return {
        category: [keyword for keyword in keywords if keyword in lower]
        for category, keywords in RISK_KEYWORDS.items()
    }


def predict_text(text):
    cleaned = clean_text(text)
    if not cleaned:
        return "BENIGN", 0.03, 0.97, "empty input"

    indicator_risk = score_with_indicator_model(text)
    if model is None or tokenizer is None:
        label = "TRAFFICKING" if indicator_risk >= 0.5 else "BENIGN"
        return label, indicator_risk, 1 - indicator_risk, "open indicator model"

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    import torch

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    transformer_risk = float(probs[1])
    risk = (transformer_risk * 0.7) + (indicator_risk * 0.3)
    safe = float(probs[0])
    return ("TRAFFICKING" if risk >= 0.5 else "BENIGN"), risk, safe, "fine-tuned transformer plus indicators"


def score_with_indicator_model(text):
    flags = detect_risk_flags(text)
    raw_score = -2.2

    for category, matches in flags.items():
        unique_matches = len(set(matches))
        raw_score += CATEGORY_WEIGHTS.get(category, 0.1) * min(unique_matches, 4) * 2.7

    lower = (text or "").lower()
    if extract_phones(text):
        raw_score += 0.35
    if extract_cities(text):
        raw_score += 0.2
    if sum(bool(matches) for matches in flags.values()) >= 3:
        raw_score += 0.8
    if "handler" in lower and ("cash only" in lower or "no police" in lower):
        raw_score += 0.9
    if ("just arrived" in lower or "new to this city" in lower) and ("village" in lower or "no family" in lower):
        raw_score += 0.6
    if any(term in lower for term in NEGATIVE_CONTEXT):
        raw_score -= 1.0

    risk = 1 / (1 + math.exp(-raw_score))
    return max(0.02, min(0.98, risk))


def risk_level(risk):
    if risk >= 0.85:
        return "Critical"
    if risk >= 0.65:
        return "High"
    if risk >= 0.5:
        return "Moderate"
    return "Low"


def analyze_record(text, threshold=0.5):
    label, risk, safe, method = predict_text(text)
    flags = detect_risk_flags(text)
    cities = extract_cities(text)
    persons = extract_persons(text)
    phones = extract_phones(text)
    flag_count = sum(len(items) for items in flags.values())

    for city in cities:
        if city in CITY_COORDS and risk >= threshold:
            city_counts[city] += 1

    return {
        "text": text,
        "label": label,
        "risk": round(risk, 4),
        "safe": round(safe, 4),
        "method": method,
        "risk_percent": round(risk * 100, 1),
        "level": risk_level(risk),
        "flagged": risk >= threshold,
        "flag_count": flag_count,
        "flags": flags,
        "phones": phones,
        "cities": cities,
        "persons": persons,
    }


def map_points():
    points = []
    max_count = max(city_counts.values()) if city_counts else 1
    for city, count in city_counts.items():
        if city in CITY_COORDS:
            lat, lng = CITY_COORDS[city]
            points.append(
                {
                    "city": city,
                    "lat": lat,
                    "lng": lng,
                    "count": count,
                    "intensity": round(count / max_count, 3),
                }
            )
    return points


def parse_batch_text(raw):
    return [line.strip() for line in (raw or "").splitlines() if line.strip()]


def parse_upload(file_storage):
    filename = (file_storage.filename or "").lower()
    content = file_storage.read().decode("utf-8", errors="ignore")

    if filename.endswith(".json"):
        data = json.loads(content)
        if isinstance(data, list):
            return [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in data]
        if isinstance(data, dict):
            return [str(value) for value in data.values()]

    if filename.endswith(".csv"):
        frame = pd.read_csv(io.StringIO(content))
        text_column = "text" if "text" in frame.columns else frame.columns[0]
        return [str(value) for value in frame[text_column].dropna().tolist()]

    rows = list(csv.reader(io.StringIO(content)))
    if len(rows) > 1 and len(rows[0]) > 1:
        return [" ".join(row).strip() for row in rows if any(cell.strip() for cell in row)]
    return parse_batch_text(content)


def summarize_results(results):
    total = len(results)
    flagged = sum(1 for item in results if item["flagged"])
    average_risk = round(sum(item["risk_percent"] for item in results) / total, 1) if total else 0
    top_cities = Counter(city for item in results for city in item["cities"] if city in CITY_COORDS)
    return {
        "total": total,
        "flagged": flagged,
        "average_risk": average_risk,
        "top_cities": top_cities.most_common(5),
    }


def fetch_public_page(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid public http or https URL.")

    response = requests.get(
        url,
        timeout=12,
        headers={"User-Agent": "Mozilla/5.0 PublicRiskResearch/1.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:12000]


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        model_loaded=MODEL_LOADED,
        demo_text=DEMO_TEXT,
    )


@app.post("/api/analyze")
def analyze_api():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    threshold = float(payload.get("threshold", 0.5))
    if not text.strip():
        return jsonify({"error": "Text is required."}), 400
    result = analyze_record(text, threshold)
    return jsonify({"result": result, "map": map_points()})


@app.post("/api/batch")
def batch_api():
    threshold = float(request.form.get("threshold", request.args.get("threshold", 0.5)))
    if "file" in request.files and request.files["file"].filename:
        texts = parse_upload(request.files["file"])
    else:
        payload = request.get_json(silent=True) or {}
        texts = parse_batch_text(payload.get("text", ""))
        threshold = float(payload.get("threshold", threshold))

    texts = texts[:200]
    if not texts:
        return jsonify({"error": "No records found."}), 400

    results = [analyze_record(text, threshold) for text in texts]
    return jsonify({"summary": summarize_results(results), "results": results, "map": map_points()})


@app.post("/api/scrape")
def scrape_api():
    payload = request.get_json(silent=True) or {}
    urls = [line.strip() for line in payload.get("urls", "").splitlines() if line.strip()]
    threshold = float(payload.get("threshold", 0.5))
    if not urls:
        return jsonify({"error": "At least one public URL is required."}), 400

    results = []
    errors = []
    for url in urls[:10]:
        try:
            text = fetch_public_page(url)
            result = analyze_record(text, threshold)
            result["source_url"] = url
            result["text"] = text[:700]
            results.append(result)
        except Exception as exc:
            errors.append({"url": url, "message": str(exc)})

    return jsonify(
        {
            "summary": summarize_results(results),
            "results": results,
            "errors": errors,
            "map": map_points(),
        }
    )


@app.get("/api/map")
def map_api():
    return jsonify({"map": map_points()})


@app.post("/api/reset-map")
def reset_map_api():
    city_counts.clear()
    return jsonify({"map": []})


@app.post("/api/demo-map")
def demo_map_api():
    threshold = float((request.get_json(silent=True) or {}).get("threshold", 0.5))
    city_counts.clear()
    results = [analyze_record(text, threshold) for text in DEMO_MAP_RECORDS]
    return jsonify({"summary": summarize_results(results), "results": results, "map": map_points()})


load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
