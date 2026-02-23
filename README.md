# 🍳📸 SnapChef — AI Recipe Reconstruction from Food Images

**SnapChef** is an end-to-end AI system that reconstructs cooking recipes from food images using a CNN–Transformer pipeline. Inspired by Meta AI’s *Inverse Cooking* research, this project demonstrates how Computer Vision and Natural Language Processing can be combined to generate ingredients and cooking instructions from a single dish image.

---

## 🚀 Overview

SnapChef takes a food image as input and performs cross-modal reasoning to generate:
* **🥕 Predicted Ingredients:** Multi-label classification of what's in the dish.
* **🍲 Cooking Instructions:** Sequential step-by-step guidance.

The system integrates:
* **ResNet-101** for high-dimensional visual feature extraction.
* **Transformer-based Attention** for alignment between visual cues and textual tokens.
* **Streamlit UI** for an interactive, real-time inference experience.

*Note: This repository focuses on an **efficient inference pipeline** and practical implementation rather than model training.*

---

## 🧠 Architecture

The system follows an "Inverse Problem" solving approach:



```mermaid
graph TD
    A[Food Image] --> B[ResNet Encoder]
    B --> C[Visual Embeddings]
    C --> D[Ingredient Predictor]
    D --> E[Predicted Ingredients]
    C & E --> F[Recipe Transformer Decoder]
    F --> G[Step-by-Step Instructions]
