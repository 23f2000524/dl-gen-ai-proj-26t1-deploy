---
title: Music Genre Classifier
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# Music Genre Classification (Messy Mashup)

This app classifies music genres from noisy mashups using a fine-tuned HuBERT model.

## Features
- Upload WAV audio
- Predict music genre
- Handles noisy and mixed audio inputs

## Model Details
- Base: superb/hubert-base-superb-ks
- Fine-tuned for 10 genre classification
- Framework: PyTorch + Transformers

## Dataset Characteristics
- Multi-stem mixing (drums, vocals, bass, others)
- Tempo alignment across tracks
- Noise injection for robustness

## Usage
Upload a 10-second WAV file and get predicted genre with confidence.