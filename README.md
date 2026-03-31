{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # \uc0\u55356 \u57269  Music Genre Classification (Messy Mashup)\
\
This app classifies music genres from noisy mashups using a fine-tuned HuBERT model.\
\
## Key Features\
- Handles noisy mixed audio inputs\
- Robust to tempo changes and multi-stem mixing\
- Predicts 10 music genres\
\
## Model\
- Base: superb/hubert-base-superb-ks\
- Fine-tuned on mashup dataset\
- Framework: PyTorch + Transformers\
\
## Dataset Characteristics\
- Multi-stem mixing (drums, vocals, bass, others)\
- Tempo alignment across tracks\
- Random noise injection\
\
This improves generalization to real-world audio.}