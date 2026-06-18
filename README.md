# Counter Trafficking Risk Dashboard

Counter Trafficking Risk Dashboard is a Flask-based web application for analyzing public text, uploaded records, and public webpages for trafficking-risk indicators. The system provides risk scoring, entity extraction, batch screening, public web text extraction, and geographic hotspot visualization through a professional HTML, CSS, and JavaScript interface.

## Preview

![Counter Trafficking Risk Dashboard](docs/result-webpage.png)

## Overview

The application helps transform unstructured public text into structured risk intelligence. It detects indicator patterns, extracts city mentions, assigns a risk score, and plots flagged locations on an interactive Leaflet map.

The system works immediately with a transparent indicator-based scoring model. It also includes an optional open-source fine-tuning pipeline using `distilroberta-base` for teams that want to train a custom classifier on labeled data.

## Key Features

- Clean white professional dashboard interface
- Single text risk analysis
- Batch scanning for TXT, CSV, and JSON files
- Public webpage text extraction and analysis
- Risk score, risk level, flag count, and city detection
- Entity extraction for cities, phone numbers, and names when spaCy is available
- Interactive Leaflet hotspot map
- Heatmap and city marker overlays
- Demo map loader with 20 sample records
- Optional DistilRoBERTa fine-tuning pipeline
- Flask JSON API backend
- No pickle model dependency

## Technology Stack

```text
Frontend: HTML, CSS, JavaScript
Backend: Flask
Mapping: Leaflet, Leaflet Heat
NLP: Indicator scoring, optional spaCy NER
Fine-tuning: Hugging Face Transformers, DistilRoBERTa
Data: CSV, JSON, TXT, public webpage text
```

## Project Structure

```text
app.py
fine_tune.py
requirements.txt
gitattributes
README.md
templates/
  index.html
static/
  styles.css
  app.js
training_data/
  sample_train.csv
  demo_map_records.csv
docs/
  result-webpage.png
models/
  trafficking-roberta
```

## Risk Scoring

The default scoring engine is a transparent indicator model. It evaluates text using weighted categories such as:

- Control language
- Movement or transit indicators
- Age or vulnerability indicators
- Isolation indicators
- Transaction language
- Secrecy or avoidance language

If a fine-tuned model is available in `models/trafficking-roberta`, the application automatically loads it and combines transformer inference with the indicator model.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch installation fails because of your Python version, install a compatible PyTorch build from the official PyTorch installation guide, then install the remaining dependencies again.

## Run the Application

```bash
python app.py
```

Open the dashboard:

```text
http://127.0.0.1:7860
```

## Demo Map Data

The project includes sample demo records for visualizing map behavior:

```text
training_data/demo_map_records.csv
```

Use the `Load demo data` button in the Hotspot Map tab to populate the map with sample city markers and heatmap points.

## Fine-Tuning

The optional fine-tuning script expects a CSV or JSON dataset with two fields:

```text
text,label
```

Supported labels:

```text
BENIGN
TRAFFICKING
```

Run fine-tuning:

```bash
python fine_tune.py --data training_data/sample_train.csv --epochs 3
```

The trained model is saved to:

```text
models/trafficking-roberta
```

Restart the Flask application after training. The backend will automatically detect and load the fine-tuned model.

## API Endpoints

```text
GET  /
POST /api/analyze
POST /api/batch
POST /api/scrape
GET  /api/map
POST /api/demo-map
POST /api/reset-map
```

## Responsible Use

This project is a prototype for lawful, authorized, public, or consented data analysis. It should not be used as the sole basis for operational decisions. Predictions are decision-support signals and require human review, validation, privacy safeguards, and domain expert oversight.

Do not upload private survivor information, confidential case files, or sensitive personal data without proper authorization.
