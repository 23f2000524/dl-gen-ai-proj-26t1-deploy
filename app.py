{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import torch\
import librosa\
import numpy as np\
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification\
\
st.title("\uc0\u55356 \u57269  Music Genre Classifier (HuBERT - Messy Mashup)")\
\
# ===== CONFIG =====\
MODEL_ID = "superb/hubert-base-superb-ks"\
SR = 16000\
DURATION = 10\
SAMPLES = SR * DURATION\
\
GENRES = [\
    "blues","classical","country","disco","hiphop",\
    "jazz","metal","pop","reggae","rock"\
]\
\
# ===== LOAD MODEL =====\
@st.cache_resource\
def load_model():\
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)\
\
    model = AutoModelForAudioClassification.from_pretrained(\
        MODEL_ID,\
        num_labels=10\
    )\
\
    # Load your trained weights\
    state_dict = torch.load("model.pt", map_location="cpu")\
    model.load_state_dict(state_dict)\
\
    model.config.id2label = \{i: g for i, g in enumerate(GENRES)\}\
    model.eval()\
\
    return model, feature_extractor\
\
model, feature_extractor = load_model()\
\
# ===== UI =====\
uploaded_file = st.file_uploader("Upload WAV audio", type=["wav"])\
\
if uploaded_file:\
    st.audio(uploaded_file)\
\
    try:\
        # ===== PREPROCESS (MATCH TRAINING) =====\
        y, _ = librosa.load(uploaded_file, sr=SR)\
\
        # Pad / crop to 10 sec\
        if len(y) < SAMPLES:\
            y = np.pad(y, (0, SAMPLES - len(y)))\
        y = y[:SAMPLES]\
\
        # Normalize\
        y = y.astype(np.float32)\
        y = y / (np.abs(y).max() + 1e-8)\
\
        # Feature extraction\
        inputs = feature_extractor(\
            y,\
            sampling_rate=SR,\
            return_tensors="pt",\
            padding=True,\
            truncation=True,\
            max_length=SAMPLES\
        )\
\
        # ===== INFERENCE =====\
        with torch.no_grad():\
            logits = model(**inputs).logits\
            probs = torch.softmax(logits, dim=-1)\
\
        pred = torch.argmax(probs, dim=-1).item()\
\
        st.success(f"\uc0\u55356 \u57263  Prediction: \{GENRES[pred]\}")\
        st.write(f"Confidence: \{probs[0][pred]:.2f\}")\
\
        # Top-3 predictions (bonus)\
        st.write("Top predictions:")\
        top3 = torch.topk(probs, 3)\
        for i in range(3):\
            st.write(\
                GENRES[top3.indices[0][i]],\
                float(top3.values[0][i])\
            )\
\
    except Exception as e:\
        st.error(str(e))}