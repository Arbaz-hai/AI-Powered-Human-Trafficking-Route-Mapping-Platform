---
title: TDS·NET — AI Trafficking Detection System
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: true
license: other
tags:
- nlp
- text-classification
- roberta
- safety
- counter-trafficking
---

# 🚨 TDS·NET — AI-Powered Human Trafficking Route Mapping Platform

> **FOR LAW ENFORCEMENT, NGO, AND AUTHORIZED COUNTER-TRAFFICKING RESEARCH USE ONLY**

---

## Overview

An AI-powered intelligence platform that leverages fine-tuned **RoBERTa NLP** to analyze
classified advertisements, phone records, and dark web text — detecting linguistic patterns
associated with human trafficking, extracting key entities, and identifying geographic hotspots
to help law enforcement agencies act faster and rescue victims.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Single Ad Analysis** | Paste any text and get instant risk scoring + entity extraction |
| 📁 **Batch File Upload** | Upload `.txt`, `.csv`, or `.json` files for bulk analysis |
| 📋 **Multi-Line Scan** | Paste multiple ads (one per line) for rapid batch screening |
| 🏙 **Hotspot Detection** | Identifies cities most frequently mentioned in high-risk ads |
| ⚠ **Risk Flag Breakdown** | Categorizes control language, movement indicators, age markers, secrecy signals |
| 🔦 **Suspicious Text Highlight** | Visually highlights risky phrases and phone numbers |

---

## Model

- **Architecture**: RoBERTa-base fine-tuned for binary sequence classification
- **Classes**: `BENIGN` (0) / `TRAFFICKING` (1)
- **Checkpoint**: `best.pt` (PyTorch state dict)
- **Training Data**: 10,200 annotated ads (balanced 50/50), preprocessed from CTDC datasets
- **Optimized for**: F1 score on trafficking class (recall-biased)

---

## File Structure

```
your-space/
├── app.py              ← Main Gradio application
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── best.pt             ← Trained model checkpoint (upload from Kaggle)
```

---

## Deployment Steps

1. Create a new Hugging Face Space → **SDK: Gradio** → Hardware: CPU Basic (free)
2. Upload `app.py`, `requirements.txt`, `README.md`
3. Upload `best.pt` from your Kaggle output (`/kaggle/working/best.pt`)
4. Space auto-builds and deploys in ~3 minutes

---

## Risk Score Interpretation

| Score | Level | Action |
|---|---|---|
| 85–100% | 🚨 CRITICAL | Escalate immediately |
| 65–84% | ⚠ HIGH | Prioritize investigation |
| 50–64% | ⚡ MODERATE | Flag for monitoring |
| < 50% | ✅ BENIGN | Low probability |

---

## Datasets Used in Training

- **CTDC Global Dataset** — 48,801 real trafficking cases (ctdatacollaborative.org)
- **CTDC Global Synthetic v2025** — 257,969 privacy-safe records
- **NLP_10k** — 10,200 synthetic annotated ads (custom generated)
- **K-Anonymized Dataset** — 97,749 de-identified records

---

## Ethical Disclaimer

This tool is designed **exclusively** for counter-trafficking operations.
Do not upload personally identifiable information of survivors or victims.
All inference is performed locally — no data is logged or stored.
Model predictions are probabilistic and must be reviewed by trained professionals
before any operational decision. Results must not be used as sole evidentiary basis.

Aligned with Polaris Project guidelines and CTDC data standards.

---

*Developed by ArbazDevHive | Powered by RoBERTa + spaCy NER*
