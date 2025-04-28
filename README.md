# Multi-modal Sentiment and Emotion Classification Using Video and Audio Cues

## Table of Contents

- [Overview](#overview)  
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Models](#models)  
  - [Emotion Recognition (GRU)](#emotion-recognition-gru)  
  - [Sentiment Recognition (LSTM)](#sentiment-recognition-lstm)  
- [Performance](#performance)  
- [Installation](#installation) 
- [Citation](#citation)  
- [License](#license) 

---

## Overview 

Speech and facial expressions carry rich affective cues—especially when text is unavailable or unreliable.  
This project fuses:  
1. **Audio** via an LSTM on Mel-spectrogram features  
2. **Video** via a CNN on facial-expression frames  

to predict simultaneously:  
- **Sentiment**: positive / neutral / negative  
- **Emotion**: anger / disgust / sadness / joy / neutral / surprise / fear  
We first perform detailed EDA to understand label distributions, conversation flows, and utterance characteristics before modeling.
---

## Features

- **Dual-label output**: Sentiment + Emotion  
- **Pretrained** on the MELD dataset for quick prototyping  

---

## Exploratory Data Analysis (EDA)

- An extensive EDA was conducted to better understand the MELD dataset before modeling.

- **Notebook:** eda-meld.ipynb

- **Goals:**

  - Analyze sentiment and emotion label distributions.

  - utterance length and conversation structures.

  - Identify class imbalance and potential data issues.

- **Findings:**
  **Sentiment Class Distribution - Training set**
    - Neutral: 4,710 instances

    - Positive: 2,945 instances

    - Negative: 2,334 instances

  **Observation:**
  - Neutral sentiment is the most common, followed by positive and negative.

  **Emotion Class Distribution - Training set**
    - Neutral: 4,710 instances

    - Joy: 1,743 instances

    - Sadness: 1,205 instances

    - Anger: 1,109 instances

    - Surprise: 683 instances

    - Fear: 271 instances

    - Disgust: 268 instances

  **Observation:**

  - Neutral emotion dominates.

  - Fear and Disgust have very few examples — leading to a strong class imbalance.

- **Impact:**

  - Insights from EDA guide model design, loss weighting, and data augmentation strategies.

---

## Models

### GRU Baseline (GRU_Audio.ipynb)

- **File**: `audio_weights_emotion.hdf5`, `audio_weights_sentiment.hdf5`
- **Architecture**: Bidirectional GRU + masking + time-distributed layers  
- **Input**: 1,611 acoustic features per time frame (Emotion), 1,422 acoustic features per time frame  (Sentiment)
- **Output**: 7 classes (neutral, joy, sadness, anger, surprise, fear, disgust), 3 classes (neutral, positive, negative) 
- **Optimizer**: Adam  

### LSTM Improved (LSTM_AUdio.ipynb)

- **File**: `.wav` convert video clips to audio files (.wav)  
- **Architecture**: Two-layer Bidirectional LSTM + masking + time-distributed layers  
- **Input**: 1,611 acoustic features per time frame (Emotion), 1,422 acoustic features per time frame  
- **Output**: 7 classes (neutral, joy, sadness, anger, surprise, fear, disgust), 3 classes (neutral, positive, negative)  
- **Optimizer**: Adadelta  

---

## Performance

| Model              | Accuracy | F1-Score |
|--------------------|---------:|---------:|
| **GRU**            |    9.17% |     0.04 |
| **LSTM (Both)**    |   48.12% |     0.31 |

---

## Installation

1. Clone this repo  
   ```bash
   git clone https://github.com/TarunSadarla2606/DL_Project
   cd DL_Project

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Start Jupyter Notebook:
  ```bash
  jupyter notebook

## Citation

S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. ACL 2019.

Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W. EmotionLines: An Emotion Corpus of Multi-Party Conversations. arXiv preprint arXiv:1802.08379 (2018).
