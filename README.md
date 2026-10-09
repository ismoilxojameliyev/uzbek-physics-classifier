# Uzbek Physics AI Classifier

An end-to-end NLP pipeline and domain classifier built to detect and categorize Uzbek-language physics problems using transfer learning on top of `xlm-roberta-base`.

## Overview
Low-resource languages such as Uzbek face dual challenges in NLP: scarce annotated domain data and script divergence (Latin vs. Cyrillic). This project bridges that gap by implementing a custom transliteration and text normalization pipeline, fine-tuning a multilingual transformer on domain-specific STEM text, and exposing the resulting model via an interactive Gradio interface.

## Pipeline Architecture
1. **Data Preprocessing & Normalization:**
   - Strips irregular whitespace and standardizes orthographic variations (*tutuq belgisi* `'` vs backticks/curly quotes).
   - Cyrillic-to-Latin transliteration mapping for morphological consistency.
2. **Model Fine-Tuning:**
   - Base Architecture: `xlm-roberta-base` (Hugging Face `transformers`)
   - Sequence length: 128 tokens with dynamic padding and truncation
   - Training: Supervised sequence classification with cross-entropy loss
3. **Inference & UI:**
   - Real-time classification interface built with Gradio and Hugging Face Pipelines.

## Project Structure
- `app.py`: Full model initialization, training pipeline, and Gradio web interface.
- `uzbek_physics_dataset.csv`: Labeled seed dataset containing physics word problems and general Uzbek text.
