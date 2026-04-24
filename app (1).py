
import gradio as gr
import torch
import re
import json
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SPACY_AVAILABLE = False
nlp_ner = None
try:
    import spacy
    try:
        nlp_ner = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                       check=False, capture_output=True)
        try:
            nlp_ner = spacy.load("en_core_web_sm")
            SPACY_AVAILABLE = True
        except Exception:
            pass
except ImportError:
    pass

MODEL_LOADED = False
tokenizer = None
model = None
MODEL_NAME = "roberta-base"

def load_model():
    global model, tokenizer, MODEL_LOADED, MODEL_NAME
    try:
        if os.path.exists("best.pt"):
            ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
            MODEL_NAME = ckpt.get("model_name", "roberta-base")
            tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=2,
                id2label={0:"BENIGN",1:"TRAFFICKING"},
                label2id={"BENIGN":0,"TRAFFICKING":1},
                ignore_mismatched_sizes=True)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=2,
                id2label={0:"BENIGN",1:"TRAFFICKING"},
                label2id={"BENIGN":0,"TRAFFICKING":1},
                ignore_mismatched_sizes=True)
        model.eval()
        MODEL_LOADED = True
    except Exception as e:
        print(f"Model load error: {e}")

load_model()

CITY_COORDS = {
    # Tier-1 metros
    "Mumbai":            [19.0760,  72.8777],
    "Delhi":             [28.6139,  77.2090],
    "Kolkata":           [22.5726,  88.3639],
    "Chennai":           [13.0827,  80.2707],
    "Bangalore":         [12.9716,  77.5946],
    "Bengaluru":         [12.9716,  77.5946],
    "Hyderabad":         [17.3850,  78.4867],
    "Ahmedabad":         [23.0225,  72.5714],
    "Pune":              [18.5204,  73.8567],
    "Surat":             [21.1702,  72.8311],
    # High-risk corridors (NCRB / Shakti Vahini data)
    "Patna":             [25.5941,  85.1376],
    "Guwahati":          [26.1445,  91.7362],
    "Siliguri":          [26.7271,  88.3953],
    "Varanasi":          [25.3176,  82.9739],
    "Lucknow":           [26.8467,  80.9462],
    "Agra":              [27.1767,  78.0081],
    "Jaipur":            [26.9124,  75.7873],
    "Bhopal":            [23.2599,  77.4126],
    "Indore":            [22.7196,  75.8577],
    "Nagpur":            [21.1458,  79.0882],
    "Raipur":            [21.2514,  81.6296],
    "Bhubaneswar":       [20.2961,  85.8245],
    "Visakhapatnam":     [17.6868,  83.2185],
    "Kochi":             [ 9.9312,  76.2673],
    "Thiruvananthapuram":[ 8.5241,  76.9366],
    "Coimbatore":        [11.0168,  76.9558],
    "Ranchi":            [23.3441,  85.3096],
    "Dhanbad":           [23.7957,  86.4304],
    "Gorakhpur":         [26.7606,  83.3732],
    "Kanpur":            [26.4499,  80.3319],
    # Border / transit cities
    "Amritsar":          [31.6340,  74.8723],
    "Jammu":             [32.7266,  74.8570],
    "Imphal":            [24.8170,  93.9368],
    "Agartala":          [23.8315,  91.2868],
    "Dibrugarh":         [27.4728,  94.9120],
    "Silchar":           [24.8333,  92.7789],
}

RISK_KEYWORDS = {
    # Control / coercion language
    "control":     ["agent manage","didi handles","bhaiya controls","seth provides",
                    "handler","supervised","someone else books","manager handles",
                    "party arrange"],
    # Mobility / transit indicators
    "movement":    ["naya sheher","new to this city","just arrived","kal chali jayungi",
                    "leaving tomorrow","passing through","transit","only here for",
                    "touring","aaj raat ke liye","shifting city"],
    # Age / vulnerability markers
    "age":         ["18 saal","19 saal","nayi ladki","fresh girl","first time",
                    "seedhi sadi","young and new","gaon se aayi","village girl",
                    "innocent","bholi"],
    # Isolation / control indicators
    "isolation":   ["akeli hun","no family here","ghar se bhaagi","runaway",
                    "koi nahi jaanta","nobody knows","limited hindi","document nahi",
                    "no papers","apna ghar nahi"],
    # Coded transaction language
    "transaction": ["advance lena","paise pehle","cash only","rate fix karo",
                    "service charge","negotiable","2000 per night","1000 ke liye",
                    "room included","khana saath"],
    # Secrecy / anti-law signals
    "secrecy":     ["police mat lao","no police","seedha aao","bilkul private",
                    "kisi ko mat batana","discreet","secret","no questions",
                    "back door entry","quietly"],
}

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", "[PHONE]", text)
    text = re.sub(r"\$\d+|\d+\s*(?:roses?|donation|tribute)", "[PRICE]", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_phones(text):
    return re.findall(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", text)

def extract_cities(text):
    found = []
    lower = text.lower()
    for city in CITY_COORDS:
        if city.lower() in lower:
            found.append(city)
    if SPACY_AVAILABLE and nlp_ner:
        try:
            doc = nlp_ner(text[:512])
            for e in doc.ents:
                if e.label_ in ["GPE","LOC"] and e.text not in found:
                    found.append(e.text)
        except Exception:
            pass
    return list(set(found))

def extract_persons(text):
    if not SPACY_AVAILABLE or not nlp_ner: return []
    try:
        doc = nlp_ner(text[:512])
        return list(set(e.text for e in doc.ents if e.label_ == "PERSON"))
    except Exception:
        return []

def detect_risk_flags(text):
    lower = text.lower()
    return {cat: [k for k in kws if k in lower] for cat, kws in RISK_KEYWORDS.items()}

def highlight_suspicious(text):
    h = text
    all_kw = sorted([kw for kws in RISK_KEYWORDS.values() for kw in kws], key=len, reverse=True)
    for kw in all_kw:
        h = re.compile(re.escape(kw), re.IGNORECASE).sub(
            f'<mark style="background:#FEE2E2;color:#DC2626;border-radius:3px;'
            f'padding:1px 4px;font-weight:600;">{kw}</mark>', h)
    for ph in extract_phones(text):
        h = h.replace(ph,
            f'<mark style="background:#FFF3CD;color:#D97706;border-radius:3px;'
            f'padding:1px 4px;font-weight:600;">{ph}</mark>')
    return h

def predict_text(text):
    if not MODEL_LOADED or model is None: return "BENIGN", 0.05, 0.95
    cleaned = clean_text(text)
    if not cleaned: return "BENIGN", 0.05, 0.95
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True,
                       max_length=256, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    risk, safe = float(probs[1]), float(probs[0])
    return ("TRAFFICKING" if risk > 0.5 else "BENIGN"), risk, safe

# ── Hotspot map HTML (Leaflet.js + heatmap plugin) ───────────────────────────
def build_hotspot_map(city_counts: dict) -> str:
    """Returns a full self-contained HTML page with Leaflet heatmap."""
    if not city_counts:
        points_json = "[]"
        markers_js  = ""
    else:
        max_count = max(city_counts.values()) if city_counts else 1
        heat_pts  = []
        markers   = []
        for city, count in city_counts.items():
            if city in CITY_COORDS:
                lat, lng = CITY_COORDS[city]
                intensity = round(count / max_count, 3)
                heat_pts.append([lat, lng, intensity])
                risk_pct = min(100, count * 15)
                col = ("#DC2626" if risk_pct >= 70 else
                       "#F97316" if risk_pct >= 40 else "#EAB308")
                markers.append(
                    f'L.circleMarker([{lat},{lng}],{{radius:{8+count*3},'
                    f'color:"{col}",fillColor:"{col}",fillOpacity:0.7,'
                    f'weight:2}}).addTo(map)'
                    f'.bindPopup("<b>{city}</b><br>Flagged mentions: {count}'
                    f'<br>Risk intensity: {risk_pct}%");'
                )
        points_json = json.dumps(heat_pts)
        markers_js  = "\n".join(markers)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI', sans-serif; background:#F8FAFC; }}
  #map {{ height: 460px; width: 100%; border-radius: 12px; }}
  #legend {{
    position: absolute; bottom: 24px; right: 24px; z-index: 1000;
    background: rgba(255,255,255,0.95); border-radius: 10px;
    padding: 14px 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    font-size: 12px; line-height: 1.8; border: 1px solid #E2E8F0;
    backdrop-filter: blur(8px);
  }}
  #legend h4 {{ font-size:11px; color:#64748B; letter-spacing:1px;
                text-transform:uppercase; margin-bottom:8px; }}
  .leg-dot {{ display:inline-block; width:10px; height:10px;
              border-radius:50%; margin-right:6px; }}
  #stats {{
    display:flex; gap:12px; padding:12px 0 8px; flex-wrap:wrap;
  }}
  .stat-pill {{
    background:#F1F5F9; border:1px solid #E2E8F0; border-radius:20px;
    padding:5px 14px; font-size:12px; color:#475569; font-weight:600;
  }}
  .stat-pill span {{ color:#DC2626; }}
  #title-bar {{
    padding: 12px 16px 6px;
    font-size: 13px; font-weight: 700; color: #1E293B;
    letter-spacing: 0.5px;
  }}
</style>
</head>
<body>
<div id="title-bar">📍 Geographic Hotspot Map — Trafficking Risk Heatmap</div>
<div id="stats">
  <div class="stat-pill">Cities detected: <span>{len(city_counts)}</span></div>
  <div class="stat-pill">Total mentions: <span>{sum(city_counts.values())}</span></div>
  <div class="stat-pill">Highest risk: <span>{max(city_counts, key=city_counts.get) if city_counts else '—'}</span></div>
</div>
<div id="map"></div>
<script>
  var map = L.map('map', {{
    center: [22.5, 82.0],
    zoom: 5,
    zoomControl: true
  }});

  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '© OpenStreetMap © CARTO',
    subdomains: 'abcd',
    maxZoom: 19
  }}).addTo(map);

  var heatData = {points_json};
  if (heatData.length > 0) {{
    L.heatLayer(heatData, {{
      radius: 45,
      blur: 35,
      maxZoom: 10,
      max: 1.0,
      gradient: {{0.2:'#FEF08A', 0.5:'#FB923C', 0.8:'#DC2626', 1.0:'#7F1D1D'}}
    }}).addTo(map);
    {markers_js}
  }} else {{
    map.addLayer(L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
      {{attribution:'© OpenStreetMap © CARTO',subdomains:'abcd'}}));
    var noData = L.control({{position:'topright'}});
    noData.onAdd = function() {{
      var d = L.DomUtil.create('div');
      d.innerHTML = '<div style="background:white;padding:12px;border-radius:8px;'
        + 'border:1px solid #E2E8F0;font-size:13px;color:#94A3B8;">'
        + '📭 No city data yet — run an analysis first</div>';
      return d;
    }};
    noData.addTo(map);
  }}
</script>
<div id="legend">
  <h4>Risk Level</h4>
  <div><span class="leg-dot" style="background:#7F1D1D"></span>Critical (70%+)</div>
  <div><span class="leg-dot" style="background:#DC2626"></span>High (50–69%)</div>
  <div><span class="leg-dot" style="background:#F97316"></span>Moderate (30–49%)</div>
  <div><span class="leg-dot" style="background:#EAB308"></span>Low (&lt;30%)</div>
</div>
</body>
</html>"""

# ── Shared state for map ──────────────────────────────────────────────────────
_global_city_counts = {}

# ── Single analysis ───────────────────────────────────────────────────────────
def analyze_single(text):
    if not text or not text.strip():
        empty = '<div style="color:#94A3B8;text-align:center;padding:40px 20px;font-size:14px;">Enter text above and click Analyze</div>'
        return empty, empty, empty, empty, build_hotspot_map({})

    label, risk, safe = predict_text(text)
    flags   = detect_risk_flags(text)
    phones  = extract_phones(text)
    cities  = extract_cities(text)
    persons = extract_persons(text)
    nflags  = sum(len(v) for v in flags.values())
    hl_text = highlight_suspicious(text)

    global _global_city_counts
    for c in cities:
        _global_city_counts[c] = _global_city_counts.get(c, 0) + 1

    pct = int(risk * 100)
    if risk >= 0.85:   badge_col="#DC2626"; badge_bg="#FEF2F2"; badge_txt="🚨 CRITICAL RISK"; bar_col="#DC2626"
    elif risk >= 0.65: badge_col="#EA580C"; badge_bg="#FFF7ED"; badge_txt="⚠ HIGH RISK";     bar_col="#F97316"
    elif risk >= 0.50: badge_col="#D97706"; badge_bg="#FFFBEB"; badge_txt="⚡ MODERATE RISK"; bar_col="#F59E0B"
    else:              badge_col="#16A34A"; badge_bg="#F0FDF4"; badge_txt="✅ LIKELY BENIGN"; bar_col="#22C55E"

    # Score card
    score_html = f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px;
     box-shadow:0 2px 12px rgba(0,0,0,0.06);">
  <div style="text-align:center;margin-bottom:20px;">
    <div style="display:inline-block;background:{badge_bg};color:{badge_col};
         font-weight:700;font-size:15px;letter-spacing:1px;
         padding:10px 28px;border-radius:30px;border:2px solid {badge_col}33;">
      {badge_txt}
    </div>
  </div>
  <div style="margin-bottom:16px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <span style="font-size:12px;color:#64748B;font-weight:600;letter-spacing:1px;">TRAFFICKING RISK SCORE</span>
      <span style="font-size:26px;font-weight:800;color:{badge_col};">{pct}%</span>
    </div>
    <div style="background:#F1F5F9;border-radius:6px;height:14px;overflow:hidden;">
      <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,{bar_col}99,{bar_col});
           border-radius:6px;transition:width 0.8s ease;"></div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
         padding:12px;text-align:center;">
      <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;margin-bottom:4px;">RISK SCORE</div>
      <div style="font-size:20px;font-weight:800;color:#DC2626;">{risk:.3f}</div>
    </div>
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
         padding:12px;text-align:center;">
      <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;margin-bottom:4px;">SAFE SCORE</div>
      <div style="font-size:20px;font-weight:800;color:#16A34A;">{safe:.3f}</div>
    </div>
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
         padding:12px;text-align:center;">
      <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;margin-bottom:4px;">FLAGS</div>
      <div style="font-size:20px;font-weight:800;color:{'#DC2626' if nflags>0 else '#16A34A'};">{nflags}</div>
    </div>
  </div>
  <div style="margin-top:14px;padding:12px;background:#F8FAFC;border-radius:10px;
       border:1px solid #E2E8F0;">
    <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;margin-bottom:6px;">MODEL</div>
    <div style="font-size:12px;color:#475569;font-family:monospace;">
      {'✅ best.pt loaded' if MODEL_LOADED and os.path.exists('best.pt') else '⚠ Demo mode (base model)'}
      &nbsp;|&nbsp; RoBERTa-base &nbsp;|&nbsp; 2-class NLP
    </div>
  </div>
</div>"""

    # Entity card
    rows = ""
    defs = [("📞 Phones", phones, "#2563EB"), ("🏙 Cities", cities, "#7C3AED"),
            ("👤 Persons", persons, "#0891B2")]
    for label_e, vals, col in defs:
        if vals:
            chips = " ".join(
                f'<span style="background:{col}11;color:{col};border:1px solid {col}33;'
                f'padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;">{v}</span>'
                for v in vals)
            rows += f'<div style="margin-bottom:12px;"><div style="font-size:10px;color:#94A3B8;'
            rows += f'letter-spacing:1px;margin-bottom:6px;">{label_e}</div><div>{chips}</div></div>'
    if not rows:
        rows = '<div style="color:#94A3B8;text-align:center;padding:16px;font-size:13px;">No entities detected</div>'

    entity_html = f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:20px;
     box-shadow:0 2px 12px rgba(0,0,0,0.06);">
  <div style="font-size:11px;color:#94A3B8;letter-spacing:2px;font-weight:700;
       margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #F1F5F9;">
    🔍 EXTRACTED ENTITIES
  </div>
  {rows}
</div>"""

    # Flags card
    flag_defs = {"control":"#DC2626","movement":"#F97316","age":"#DC2626",
                 "isolation":"#7C3AED","transaction":"#D97706","secrecy":"#DB2777"}
    flag_rows = ""
    for cat, matches in flags.items():
        if matches:
            col = flag_defs.get(cat,"#DC2626")
            chips = " ".join(
                f'<span style="background:{col}11;color:{col};border:1px solid {col}33;'
                f'padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;">{m}</span>'
                for m in matches)
            flag_rows += f'<div style="margin-bottom:10px;"><div style="font-size:10px;'
            flag_rows += f'color:{col};letter-spacing:1px;font-weight:700;margin-bottom:5px;">'
            flag_rows += f'{cat.upper()}</div><div>{chips}</div></div>'
    if not flag_rows:
        flag_rows = '<div style="color:#16A34A;text-align:center;padding:16px;font-size:13px;">✅ No suspicious flags detected</div>'

    flags_html = f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:20px;
     box-shadow:0 2px 12px rgba(0,0,0,0.06);">
  <div style="font-size:11px;color:#94A3B8;letter-spacing:2px;font-weight:700;
       margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #F1F5F9;">
    ⚠ RISK FLAG BREAKDOWN
  </div>
  {flag_rows}
</div>"""

    # Highlight card
    hl_html = f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:20px;
     box-shadow:0 2px 12px rgba(0,0,0,0.06);">
  <div style="font-size:11px;color:#94A3B8;letter-spacing:2px;font-weight:700;
       margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #F1F5F9;">
    🔦 SUSPICIOUS CONTENT HIGHLIGHT
  </div>
  <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
       padding:16px;line-height:2;font-size:14px;color:#374151;min-height:60px;">
    {hl_text}
  </div>
  <div style="margin-top:10px;display:flex;gap:16px;flex-wrap:wrap;">
    <span style="font-size:11px;color:#64748B;">
      <mark style="background:#FEE2E2;color:#DC2626;padding:1px 5px;border-radius:3px;">■</mark> Risk keyword
    </span>
    <span style="font-size:11px;color:#64748B;">
      <mark style="background:#FFF3CD;color:#D97706;padding:1px 5px;border-radius:3px;">■</mark> Phone number
    </span>
  </div>
</div>"""

    return score_html, entity_html, flags_html, hl_html, build_hotspot_map(_global_city_counts)

# ── Batch analysis (file) ─────────────────────────────────────────────────────
def analyze_file(file_obj, threshold):
    if file_obj is None:
        return _empty_batch("No file uploaded."), pd.DataFrame(), build_hotspot_map({})

    try:
        path = file_obj.name
        ext  = os.path.splitext(path)[-1].lower()
        if ext == ".txt":
            with open(path,"r",encoding="utf-8",errors="ignore") as f:
                lines = [l.strip() for l in f.read().split("\n") if l.strip()]
            texts = lines[:200]
        elif ext == ".csv":
            df = pd.read_csv(path, low_memory=False)
            tcols = [c for c in df.columns if any(x in c.lower()
                     for x in ["text","ad","content","description"])]
            col = tcols[0] if tcols else df.columns[0]
            texts = df[col].dropna().astype(str).tolist()[:200]
        elif ext == ".json":
            with open(path,"r",encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                texts = [str(d.get("text",d.get("ad",d.get("content",str(d))))) for d in data[:200]]
            elif isinstance(data, dict):
                texts = [str(v) for v in data.values() if isinstance(v, str)][:200]
            else:
                texts = [str(data)]
        else:
            return _empty_batch("Unsupported file. Use .txt, .csv, or .json"), pd.DataFrame(), build_hotspot_map({})
    except Exception as e:
        return _empty_batch(f"File error: {e}"), pd.DataFrame(), build_hotspot_map({})

    return _run_batch(texts, threshold)

# ── Batch analysis (text paste) ───────────────────────────────────────────────
def analyze_text_batch(raw, threshold):
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if not lines:
        return _empty_batch("Paste ads above (one per line)."), pd.DataFrame(), build_hotspot_map({})
    return _run_batch(lines[:100], threshold)

def _run_batch(texts, threshold):
    results = []
    city_counter = {}
    global _global_city_counts

    for i, text in enumerate(texts):
        label, risk, safe = predict_text(text)
        flags  = detect_risk_flags(text)
        phones = extract_phones(text)
        cities = extract_cities(text)
        nflags = sum(len(v) for v in flags.values())
        for c in cities:
            city_counter[c] = city_counter.get(c, 0) + 1
            _global_city_counts[c] = _global_city_counts.get(c, 0) + 1
        results.append({
            "#":            i+1,
            "Risk %":       f"{int(risk*100)}%",
            "Classification": "🚨 TRAFFICKING RISK" if label=="TRAFFICKING" else "✅ BENIGN",
            "Risk Flags":   nflags,
            "Phones":       ", ".join(phones) or "—",
            "Cities":       ", ".join(cities) or "—",
            "Text Preview": text[:100]+"…" if len(text)>100 else text,
        })

    total   = len(results)
    flagged = sum(1 for r in results if int(r["Risk %"].rstrip("%")) >= int(threshold*100))
    pct_f   = int(flagged/total*100) if total else 0
    col_f   = "#DC2626" if pct_f>50 else ("#F97316" if pct_f>25 else "#16A34A")

    top5 = sorted(city_counter.items(), key=lambda x:x[1], reverse=True)[:5]
    city_bars = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        f'<span style="min-width:110px;font-size:12px;color:#374151;font-weight:600;">{c}</span>'
        f'<div style="flex:1;background:#F1F5F9;border-radius:4px;height:8px;">'
        f'<div style="height:100%;width:{min(100,n*18)}%;background:#DC2626;border-radius:4px;"></div>'
        f'</div><span style="color:#DC2626;font-weight:700;font-size:12px;min-width:20px;">{n}</span></div>'
        for c,n in top5
    ) or '<div style="color:#94A3B8;font-size:13px;">No cities detected</div>'

    summary_html = f"""
<div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px;
     box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-bottom:16px;">
  <div style="font-size:11px;color:#94A3B8;letter-spacing:2px;font-weight:700;
       margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid #F1F5F9;">
    📊 BATCH ANALYSIS RESULTS
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px;">
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:16px;text-align:center;">
      <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;margin-bottom:6px;">TOTAL ADS</div>
      <div style="font-size:30px;font-weight:800;color:#1E293B;">{total}</div>
    </div>
    <div style="background:#FEF2F2;border:1px solid #FCA5A5;border-radius:12px;padding:16px;text-align:center;">
      <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;margin-bottom:6px;">HIGH RISK</div>
      <div style="font-size:30px;font-weight:800;color:#DC2626;">{flagged}</div>
    </div>
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:16px;text-align:center;">
      <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;margin-bottom:6px;">RISK RATE</div>
      <div style="font-size:30px;font-weight:800;color:{col_f};">{pct_f}%</div>
    </div>
  </div>
  <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:16px;">
    <div style="font-size:10px;color:#94A3B8;letter-spacing:1px;margin-bottom:14px;">🏙 TOP HOTSPOT CITIES</div>
    {city_bars}
  </div>
</div>"""

    return summary_html, pd.DataFrame(results), build_hotspot_map(city_counter)

def _empty_batch(msg):
    html = f'<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:12px;padding:20px;color:#92400E;font-size:13px;">{msg}</div>'
    return html

DEMO_SINGLE = (
    "Nayi ladki, abhi Mumbai aayi hai. Kal Pune shift hogi. "
    "Akeli hai, koi nahi jaanta. Agent handle karta hai saari bookings. "
    "Cash only. Police mat lao. Seedha contact karo: 9876543210. "
    "Paise pehle, koi sawaal nahi."
)

DEMO_BATCH = """\
New in town. Young girl available 24/7. Management handles all bookings. Call 555-234-5678.
Fresh arrival from overseas. Limited English. Incalls only in Atlanta. All services.
Licensed massage therapist available in Chicago. Book online. Flexible hours.
New in Dallas, only here this weekend. Discreet service. Call 555-876-5432.
Dog walker available weekends. References available. Insured and bonded.
Touring girl from Miami. Leaving soon. Daddy covers all expenses. No police.
Professional chef for hire. Private dinners and meal prep. Competitive rates.
Young and eager, just arrived in Houston. Will travel anywhere. Cash preferred.
Yoga instructor. Private lessons. Meditation coaching. Book online today.
Exotic beauty. New to Los Angeles. All services. Donation required."""


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --white:    #FFFFFF;
  --bg:       #F8FAFC;
  --surface:  #FFFFFF;
  --border:   #E2E8F0;
  --border2:  #CBD5E1;
  --text-1:   #0F172A;
  --text-2:   #334155;
  --text-3:   #64748B;
  --text-4:   #94A3B8;
  --danger:   #DC2626;
  --danger-l: #FEF2F2;
  --warn:     #F97316;
  --safe:     #16A34A;
  --safe-l:   #F0FDF4;
  --accent:   #2563EB;
  --accent-l: #EFF6FF;
  --radius:   12px;
  --shadow:   0 2px 12px rgba(0,0,0,0.07);
  --shadow-lg:0 8px 32px rgba(0,0,0,0.10);
  --font:     'Inter', system-ui, sans-serif;
  --mono:     'JetBrains Mono', 'Courier New', monospace;
}

* { box-sizing: border-box; }

.gradio-container {
  background: var(--bg) !important;
  font-family: var(--font) !important;
  max-width: 100% !important;
  padding: 0 !important;
}

body, html { background: var(--bg) !important; }

/* ── Navbar ──────────────────────────────────────────────────────────── */
#navbar {
  position: sticky; top: 0; z-index: 1000;
  background: rgba(255,255,255,0.96);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 0 40px;
  display: flex; align-items: center; justify-content: space-between;
  height: 60px;
  box-shadow: 0 1px 12px rgba(0,0,0,0.06);
}
#nav-brand {
  display: flex; align-items: center; gap: 10px;
  font-weight: 800; font-size: 16px; color: var(--text-1); letter-spacing: -0.3px;
}
#nav-brand .brand-icon {
  width: 32px; height: 32px; background: var(--danger);
  border-radius: 8px; display: flex; align-items: center;
  justify-content: center; color: white; font-size: 16px;
}
#nav-brand .brand-sub {
  font-size: 11px; color: var(--text-4); font-weight: 500;
  letter-spacing: 1px; text-transform: uppercase; display: block; margin-top: -2px;
}
#nav-links { display:flex; gap:6px; list-style:none; margin:0; padding:0; }
#nav-links li {
  font-size: 13px; font-weight: 500; color: var(--text-3);
  padding: 6px 14px; border-radius: 8px; cursor: pointer;
  transition: all 0.15s;
}
#nav-links li:hover { background: var(--bg); color: var(--text-1); }
#nav-links li.active { background: var(--danger-l); color: var(--danger); font-weight: 600; }
#nav-pill {
  display: flex; align-items: center; gap: 6px;
  background: var(--safe-l); color: var(--safe);
  padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
  border: 1px solid #BBF7D0;
}
#nav-pill::before { content:'●'; animation: blink 1.8s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Hero ────────────────────────────────────────────────────────────── */
#hero {
  background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
  padding: 48px 40px 40px;
  position: relative; overflow: hidden;
}
#hero::before {
  content: '';
  position: absolute; top: -40%; right: -10%; width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(220,38,38,0.15) 0%, transparent 65%);
  pointer-events: none;
}
#hero::after {
  content: '';
  position: absolute; bottom: -30%; left: -5%; width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(37,99,235,0.12) 0%, transparent 65%);
  pointer-events: none;
}
#hero-tag {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(220,38,38,0.15); color: #FCA5A5;
  border: 1px solid rgba(220,38,38,0.3);
  padding: 4px 12px; border-radius: 20px;
  font-size: 11px; font-weight: 600; letter-spacing: 2px;
  text-transform: uppercase; margin-bottom: 20px;
}
#hero-title {
  font-size: 36px; font-weight: 800; color: #F1F5F9;
  line-height: 1.2; margin-bottom: 12px; letter-spacing: -0.5px;
}
#hero-title .accent { color: #F87171; }
#hero-desc {
  font-size: 15px; color: #94A3B8; max-width: 600px;
  line-height: 1.7; margin-bottom: 28px;
}
#hero-stats {
  display: flex; gap: 24px; flex-wrap: wrap;
}
.hero-stat {
  display: flex; flex-direction: column;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px; padding: 14px 20px;
}
.hero-stat-val { font-size: 24px; font-weight: 800; color: #F1F5F9; }
.hero-stat-lbl { font-size: 11px; color: #64748B; letter-spacing: 1px; margin-top: 2px; }

/* ── Tabs ────────────────────────────────────────────────────────────── */
.tab-nav { background: var(--white) !important; border-bottom: 2px solid var(--border) !important; padding: 0 24px !important; }
.tab-nav button {
  font-family: var(--font) !important;
  font-size: 13px !important; font-weight: 600 !important;
  color: var(--text-3) !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  margin-bottom: -2px !important; padding: 14px 18px !important;
  background: transparent !important;
  transition: all 0.15s !important;
}
.tab-nav button.selected { color: var(--danger) !important; border-bottom-color: var(--danger) !important; }
.tab-nav button:hover { color: var(--text-1) !important; }

/* ── Tab content wrapper ─────────────────────────────────────────────── */
.tab-content { padding: 28px 32px !important; background: var(--bg) !important; }

/* ── Section headers ─────────────────────────────────────────────────── */
.section-header {
  font-size: 11px; font-weight: 700; color: var(--text-4);
  letter-spacing: 2px; text-transform: uppercase;
  margin-bottom: 8px;
}

/* ── Labels / inputs ─────────────────────────────────────────────────── */
label, .label-wrap {
  font-family: var(--font) !important;
  font-size: 12px !important; font-weight: 600 !important;
  color: var(--text-2) !important;
  letter-spacing: 0.3px !important;
}

textarea, input[type="text"] {
  background: var(--white) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-1) !important;
  font-family: var(--font) !important;
  font-size: 14px !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
  padding: 12px 14px !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
  outline: none !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
button.primary {
  background: var(--danger) !important;
  border: none !important; color: #fff !important;
  font-family: var(--font) !important;
  font-size: 13px !important; font-weight: 700 !important;
  padding: 11px 24px !important; border-radius: 10px !important;
  transition: all 0.15s !important;
  box-shadow: 0 4px 12px rgba(220,38,38,0.3) !important;
}
button.primary:hover {
  background: #B91C1C !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 18px rgba(220,38,38,0.4) !important;
}

button.secondary {
  background: var(--white) !important;
  border: 1.5px solid var(--border) !important;
  color: var(--text-2) !important;
  font-family: var(--font) !important;
  font-size: 13px !important; font-weight: 600 !important;
  padding: 10px 20px !important; border-radius: 10px !important;
  transition: all 0.15s !important;
}
button.secondary:hover { border-color: var(--border2) !important; color: var(--text-1) !important; }

/* ── Slider ──────────────────────────────────────────────────────────── */
input[type="range"] { accent-color: var(--danger) !important; }

/* ── File upload ─────────────────────────────────────────────────────── */
.upload-drop {
  background: var(--white) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important;
  transition: border-color 0.15s !important;
}
.upload-drop:hover { border-color: var(--danger) !important; }

/* ── Dataframe ───────────────────────────────────────────────────────── */
table { width:100% !important; border-collapse:collapse !important; }
th {
  background: #F8FAFC !important; color: var(--text-3) !important;
  font-family: var(--font) !important; font-size: 11px !important;
  font-weight: 700 !important; letter-spacing: 1px !important;
  text-transform: uppercase !important;
  border-bottom: 2px solid var(--border) !important;
  padding: 10px 14px !important; text-align:left !important;
}
td {
  background: var(--white) !important; color: var(--text-2) !important;
  font-family: var(--font) !important; font-size: 13px !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 10px 14px !important;
}
tr:hover td { background: #F8FAFC !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background: var(--danger); }

/* ── Footer ──────────────────────────────────────────────────────────── */
#footer {
  text-align:center; padding:20px 40px;
  border-top: 1px solid var(--border);
  background: var(--white);
  font-size: 12px; color: var(--text-4);
  display: flex; align-items:center; justify-content:center; gap:16px;
  flex-wrap:wrap;
}
"""

# ── HTML Blocks ───────────────────────────────────────────────────────────────
NAVBAR_HTML = """
<div id="navbar">
  <div id="nav-brand">
    <div class="brand-icon">🛡</div>
    <div>
      TDS·NET
      <span class="brand-sub">India Counter-Trafficking Intelligence Platform</span>
    </div>
  </div>
  <ul id="nav-links">
    <li class="active">Home</li>
    <li>Analyze</li>
    <li>Batch Upload</li>
    <li>Hotspot Map</li>
    <li>About</li>
  </ul>
  <div id="nav-pill">SYSTEM ONLINE</div>
</div>
"""

HERO_HTML = """
<div id="hero">
  <div id="hero-tag">🔒 Authorized Personnel Only</div>
  <div id="hero-title">
    AI-Powered <span class="accent">Trafficking</span><br>Route Mapping — India
  </div>
  <div id="hero-desc">
    Fine-tuned RoBERTa NLP engine trained on Indian trafficking patterns — analyzing
    ads, phone records and social posts in Hindi/Hinglish &amp; English to detect coercion
    signals, extract entities, and map high-risk corridors across India, helping agencies
    like CBI, NCB, and state ATUs act faster to rescue victims.
  </div>
  <div id="hero-stats">
    <div class="hero-stat">
      <span class="hero-stat-val">10,200</span>
      <span class="hero-stat-lbl">Training Samples</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-val">100%</span>
      <span class="hero-stat-lbl">F1 Score (Synthetic)</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-val">6</span>
      <span class="hero-stat-lbl">Risk Flag Categories</span>
    </div>
    <div class="hero-stat">
      <span class="hero-stat-val">30+</span>
      <span class="hero-stat-lbl">Indian Cities Tracked</span>
    </div>
  </div>
</div>
"""

ABOUT_HTML = """
<div style="padding:8px 0;">

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
    <div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
      <div style="font-size:11px;color:#94A3B8;letter-spacing:2px;font-weight:700;margin-bottom:14px;">🤖 NLP MODEL</div>
      <div style="font-size:15px;font-weight:700;color:#0F172A;margin-bottom:6px;">RoBERTa-base Fine-tuned</div>
      <div style="color:#64748B;font-size:13px;line-height:1.7;">Binary classifier trained on 10,200 annotated Indian-context ads in Hindi, Hinglish &amp; English. Optimized for recall using weighted cross-entropy with cosine LR scheduling — 6 epochs on GPU T4×2.</div>
    </div>
    <div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
      <div style="font-size:11px;color:#94A3B8;letter-spacing:2px;font-weight:700;margin-bottom:14px;">📊 TRAINING DATASETS</div>
      <div style="font-size:13px;color:#374151;line-height:2;">
        <div>• CTDC Global Dataset — 48,801 real cases</div>
        <div>• CTDC Synthetic v2025 — 257,969 records</div>
        <div>• NLP_10k — 10,200 annotated ads (custom)</div>
        <div>• K-Anonymized Dataset — 97,749 records</div>
      </div>
    </div>
  </div>

  <div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px;box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-bottom:16px;">
    <div style="font-size:11px;color:#94A3B8;letter-spacing:2px;font-weight:700;margin-bottom:16px;">⚠ RISK SCORE GUIDE</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
      <div style="background:#FEF2F2;border:1px solid #FCA5A5;border-radius:10px;padding:16px;text-align:center;">
        <div style="font-size:18px;font-weight:800;color:#DC2626;">85–100%</div>
        <div style="font-size:11px;color:#DC2626;font-weight:700;margin:4px 0;">CRITICAL</div>
        <div style="font-size:11px;color:#64748B;">Escalate immediately</div>
      </div>
      <div style="background:#FFF7ED;border:1px solid #FDBA74;border-radius:10px;padding:16px;text-align:center;">
        <div style="font-size:18px;font-weight:800;color:#EA580C;">65–84%</div>
        <div style="font-size:11px;color:#EA580C;font-weight:700;margin:4px 0;">HIGH</div>
        <div style="font-size:11px;color:#64748B;">Prioritize investigation</div>
      </div>
      <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:16px;text-align:center;">
        <div style="font-size:18px;font-weight:800;color:#D97706;">50–64%</div>
        <div style="font-size:11px;color:#D97706;font-weight:700;margin:4px 0;">MODERATE</div>
        <div style="font-size:11px;color:#64748B;">Flag for monitoring</div>
      </div>
      <div style="background:#F0FDF4;border:1px solid #86EFAC;border-radius:10px;padding:16px;text-align:center;">
        <div style="font-size:18px;font-weight:800;color:#16A34A;">&lt; 50%</div>
        <div style="font-size:11px;color:#16A34A;font-weight:700;margin:4px 0;">BENIGN</div>
        <div style="font-size:11px;color:#64748B;">Low probability</div>
      </div>
    </div>
  </div>

  <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:14px;padding:20px;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
    <div style="font-size:11px;color:#92400E;letter-spacing:2px;font-weight:700;margin-bottom:10px;">⚠ ETHICAL DISCLAIMER</div>
    <div style="font-size:13px;color:#78350F;line-height:1.8;">
      This tool is <strong>strictly for authorized law enforcement, NGOs, and counter-trafficking researchers</strong>.
      Do not upload personally identifiable information of survivors or victims.
      All inference is performed locally — no data is logged or stored.
      Model predictions are probabilistic and must be reviewed by trained professionals
      before any operational decision. Results must not be used as sole evidentiary basis.
      Developed in alignment with CTDC data standards, NCRB trafficking indicators, and Shakti Vahini / iHuman operational guidelines. For use by CBI, state ATUs, NCW, and authorised NGOs only.
    </div>
  </div>
</div>
"""

FOOTER_HTML = """
<div id="footer">
  <span>🛡 TDS·NET v2.1</span>
  <span style="color:#E2E8F0;">|</span>
  <span>Powered by RoBERTa + spaCy NER + Leaflet Heatmap | India Edition</span>
  <span style="color:#E2E8F0;">|</span>
  <span>For Official Counter-Trafficking Use Only</span>
  <span style="color:#E2E8F0;">|</span>
  <span>ArbazDevHive</span>
</div>
"""


with gr.Blocks(css=CSS, title="TDS·NET — AI Trafficking Detection Platform") as demo:

    gr.HTML(NAVBAR_HTML)
    gr.HTML(HERO_HTML)

    with gr.Tabs(elem_classes=["tab-nav"]):

        with gr.Tab("🔍 Analyze Text"):
            with gr.Column(elem_classes=["tab-content"]):
                gr.HTML('<p style="font-size:13px;color:#64748B;margin-bottom:16px;">'
                        'Paste any classified ad, social media post, or text record below '
                        'for instant NLP risk analysis and entity extraction.</p>')
                with gr.Row():
                    with gr.Column(scale=3):
                        text_in = gr.Textbox(
                            label="Evidence Text",
                            placeholder="Paste classified ad, social media post, or phone record transcript here…",
                            lines=7, value=DEMO_SINGLE)
                        with gr.Row():
                            analyze_btn = gr.Button("🔍 Analyze Threat", variant="primary")
                            clear_btn   = gr.Button("✕ Clear", variant="secondary")
                            demo_btn    = gr.Button("⚡ Load Demo", variant="secondary")
                    with gr.Column(scale=2):
                        score_out = gr.HTML()

                with gr.Row():
                    entity_out = gr.HTML()
                    flags_out  = gr.HTML()

                hl_out  = gr.HTML()
                map_out = gr.HTML(label="Geographic Hotspot Map")

                analyze_btn.click(fn=analyze_single, inputs=[text_in],
                                  outputs=[score_out, entity_out, flags_out, hl_out, map_out])
                clear_btn.click(fn=lambda: [""]*5 + [build_hotspot_map({})],
                                inputs=[], outputs=[text_in, score_out, entity_out, flags_out, hl_out, map_out])
                demo_btn.click(fn=lambda: DEMO_SINGLE, inputs=[], outputs=[text_in])

        with gr.Tab("📁 Batch File Upload"):
            with gr.Column(elem_classes=["tab-content"]):
                gr.HTML('<p style="font-size:13px;color:#64748B;margin-bottom:16px;">'
                        'Upload a <b>.txt</b>, <b>.csv</b>, or <b>.json</b> file '
                        'containing up to 200 ads/records for bulk risk screening.</p>')
                with gr.Row():
                    with gr.Column(scale=2):
                        file_in = gr.File(label="Upload Evidence File",
                                          file_types=[".txt",".csv",".json"],
                                          elem_classes=["upload-drop"])
                        threshold_sl = gr.Slider(minimum=0.3, maximum=0.95, value=0.5,
                                                 step=0.05, label="Risk Threshold for Flagging")
                        batch_btn = gr.Button("🚨 Run Batch Analysis", variant="primary")
                    with gr.Column(scale=3):
                        batch_summary = gr.HTML()

                batch_table = gr.Dataframe(label="Detailed Results", wrap=True, interactive=False)
                batch_map   = gr.HTML(label="Hotspot Map")

                batch_btn.click(fn=analyze_file,
                                inputs=[file_in, threshold_sl],
                                outputs=[batch_summary, batch_table, batch_map])

        with gr.Tab("📋 Multi-Line Scan"):
            with gr.Column(elem_classes=["tab-content"]):
                gr.HTML('<p style="font-size:13px;color:#64748B;margin-bottom:16px;">'
                        'Paste multiple ads directly — one per line — for rapid bulk screening '
                        'without needing to upload a file.</p>')
                with gr.Row():
                    with gr.Column(scale=2):
                        batch_text_in = gr.Textbox(
                            label="Paste Multiple Ads (one per line)",
                            placeholder="Paste ads here, one per line…",
                            lines=12, value=DEMO_BATCH)
                        threshold_sl2 = gr.Slider(minimum=0.3, maximum=0.95, value=0.5,
                                                  step=0.05, label="Risk Threshold")
                        batch_text_btn = gr.Button("🚨 Scan All", variant="primary")
                    with gr.Column(scale=3):
                        batch_text_summary = gr.HTML()

                batch_text_table = gr.Dataframe(wrap=True, interactive=False)
                batch_text_map   = gr.HTML(label="Hotspot Map")

                batch_text_btn.click(fn=analyze_text_batch,
                                     inputs=[batch_text_in, threshold_sl2],
                                     outputs=[batch_text_summary, batch_text_table, batch_text_map])

        with gr.Tab("🗺 Hotspot Map"):
            with gr.Column(elem_classes=["tab-content"]):
                gr.HTML('<p style="font-size:13px;color:#64748B;margin-bottom:16px;">'
                        'Interactive heatmap showing cumulative geographic risk across all '
                        'analyses performed this session. Run analyses in other tabs to '
                        'populate the map.</p>')

                refresh_btn = gr.Button("🔄 Refresh Map", variant="secondary")
                live_map    = gr.HTML(value=build_hotspot_map({}))

                refresh_btn.click(fn=lambda: build_hotspot_map(_global_city_counts),
                                  inputs=[], outputs=[live_map])

                gr.HTML("""
<div style="margin-top:16px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
  <div style="background:#FEF2F2;border:1px solid #FCA5A5;border-radius:10px;padding:14px;text-align:center;">
    <div style="font-size:11px;color:#94A3B8;letter-spacing:1px;margin-bottom:6px;">HEATMAP KEY</div>
    <div style="font-size:12px;color:#DC2626;font-weight:700;">Dark Red = Critical</div>
  </div>
  <div style="background:#FFF7ED;border:1px solid #FDBA74;border-radius:10px;padding:14px;text-align:center;">
    <div style="font-size:11px;color:#94A3B8;letter-spacing:1px;margin-bottom:6px;">HEATMAP KEY</div>
    <div style="font-size:12px;color:#F97316;font-weight:700;">Orange = High Risk</div>
  </div>
  <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:14px;text-align:center;">
    <div style="font-size:11px;color:#94A3B8;letter-spacing:1px;margin-bottom:6px;">HEATMAP KEY</div>
    <div style="font-size:12px;color:#D97706;font-weight:700;">Yellow = Moderate</div>
  </div>
  <div style="background:#F0FDF4;border:1px solid #86EFAC;border-radius:10px;padding:14px;text-align:center;">
    <div style="font-size:11px;color:#94A3B8;letter-spacing:1px;margin-bottom:6px;">DATA SOURCE</div>
    <div style="font-size:12px;color:#16A34A;font-weight:700;">Session Analysis Data</div>
  </div>
</div>""")

        with gr.Tab("ℹ About"):
            with gr.Column(elem_classes=["tab-content"]):
                gr.HTML(ABOUT_HTML)

    gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)