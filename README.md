# Multi-modal Sentiment and Emotion Classification Using Video and Audio Cues

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Models](#models)  
  - [Emotion Recognition (GRU)](#emotion-recognition-gru)  
  - [Sentiment Recognition (LSTM)](#sentiment-recognition-lstm)  
- [Performance](#performance)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Feature Extraction](#feature-extraction)  
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

---

## Features

- **Dual-label output**: Sentiment + Emotion  
- **Pretrained** on the MELD dataset for quick prototyping  

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
   cd your-repo