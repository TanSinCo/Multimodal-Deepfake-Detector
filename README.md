
# Multimodal Deepfake Detection System

## Overview

This project is a multimodal AI-based deepfake detection system designed for robust video forgery analysis using:

- Spatial facial features
- Temporal inconsistencies
- Physiological rPPG signals
- Eye blink analysis
- Facial micro-motion analysis

The system was developed as a final-semester deep learning and computer vision project with a focus on engineering depth, multimodal fusion, and practical AI pipeline development.

---

# Features

## Multimodal Detection Pipeline

The system combines multiple modalities:

1. RGB Spatial Features
2. Temporal Sequence Learning
3. Remote Photoplethysmography (rPPG)
4. Blink Signal Analysis
5. Facial Motion Signal Analysis

---

# Architecture

Video
→ Frame Extraction
→ Face Detection
→ MediaPipe Face Mesh
→ ROI Extraction
→ Signal Processing
→ Multimodal Feature Fusion
→ Deepfake Classification

---

# Models Used

## Spatial Encoder
- CNN-based facial feature extraction

## Temporal Modeling
- LSTM sequence learning

## rPPG Network
- Conv1D physiological signal encoder

## Signal Encoders
- Blink signal encoder
- Motion signal encoder

## Final Fusion Model
- Multimodal feature fusion
- Binary classification head

---

# Dataset

- FaceForensics++ style dataset
- Real/Fake video classification
- ~2000 source videos
- Compressed video setting

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Matplotlib
- CUDA GPU Training

---

# Results

## Final Metrics

- Accuracy: ~79%
- F1 Score: ~0.80

---

# Project Structure

src/
│
├── models/
├── preprocess/
├── final_train.py
├── final_inference.py
├── evaluate.py
│
results/
│
├── confusion_matrix.png
├── roc_curve.png

---

# Key Engineering Challenges Solved

- Variable sequence length handling
- MediaPipe preprocessing failures
- Multimodal tensor synchronization
- PyTorch serialization issues
- Large-scale cache preprocessing
- Memory optimization on Kaggle
- Corrupted cache recovery
- GPU-compatible training pipeline

---

# Future Improvements

- Vision Transformer integration
- Audio-visual fusion
- Real-time webcam inference
- Attention-based fusion
- Deployment with Streamlit/Gradio

---

# Author

Final Semester Deep Learning & Computer Vision Project
