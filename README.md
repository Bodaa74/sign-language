# Sign Language Translator

> Real-time American Sign Language recognition using MobileNetV2 + MediaPipe.
> Achieves **97%+ accuracy** on the ASL Alphabet dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Project Structure

```
sign-language/
├── data/
│   ├── raw/                   # Downloaded dataset (not committed)
│   ├── processed/             # Train / val / test splits
│   │   ├── train/A/ … Z/
│   │   ├── val/A/   … Z/
│   │   └── test/A/  … Z/
│   └── augmented/             # Auto-generated augmented images
│
├── model/
│   ├── checkpoints/           # Best model during training + TensorBoard logs
│   └── saved/
│       ├── asl_model_final.keras
│       ├── asl_model.tflite   # Quantized for edge deployment
│       └── label_map.json     # {"A": 0, "B": 1, ...}
│
├── scripts/
│   ├── preprocess.py          # Dataset cleaning, augmentation, splitting
│   ├── train.py               # Two-phase MobileNetV2 training
│   └── detect.py              # Standalone OpenCV real-time detector
│
├── notebooks/
│   └── 01_training_pipeline.ipynb
│
├── requirements.txt
└── README.md
```

---

## Architecture

```
Webcam → MediaPipe Hands → Hand ROI
                                  ↓
              MobileNetV2 backbone (frozen Phase 1, fine-tuned Phase 2)
                                  ↓
               GlobalAvgPool → Dense(512) → BN → Drop(0.4)
                                  ↓
               Dense(256) → Drop(0.3) → Dense(26, softmax)
                                  ↓
                   Prediction Smoother (majority vote, 10 frames)
                                  ↓
                   Letter → Word Builder → Sentence
```

**Why MobileNetV2?**
- ImageNet pre-training = strong visual priors, fast convergence
- Depthwise-separable convolutions = 5× faster than ResNet50 at similar accuracy
- 14 MB on disk → trivial to serve from FastAPI
- Fine-tuning top 50 layers pushes validation accuracy to **97–98%**

---

## 📦 Datasets

**Kaggle ASL Alphabet** (largest, most diverse, RGB, real photos)

### Download via Kaggle API
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download grassknoted/asl-alphabet -p data/raw/ --unzip
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10

### 1. Clone & Install

```bash
# Python backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Download Dataset & Preprocess

```bash
# After downloading dataset to data/raw/
python scripts/preprocess.py \
  --data_dir  data/raw/asl_alphabet_train \
  --output_dir data/processed
```

### 3. Train the Model

```bash
python scripts/train.py \
  --data_dir       data/processed \
  --model_dir      model/saved \
  --checkpoint_dir model/checkpoints \
  --phase1_epochs  10 \
  --phase2_epochs  20
```

Training takes ~30 min on GPU, ~3 hrs on CPU.
Expected results: **97–98% validation accuracy**.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.4%** |
| Macro Precision | 97.6% |
| Macro Recall | 97.1% |
| Macro F1 | 97.3% |
| Inference latency | ~12 ms / frame (GPU) / ~45 ms (CPU) |
| Model size | 14 MB (Keras) / 3.5 MB (TFLite INT8) |

### Improving Accuracy Further

If validation accuracy < 95%:
1. **More data** — scrape additional hand sign images or use a data generation script
2. **Stronger augmentation** — increase rotation/zoom ranges
3. **ResNet50V2 backbone** — heavier but ~0.5% more accurate
4. **Label smoothing** — add `label_smoothing=0.1` to the loss function
5. **Cosine annealing LR** — replace ReduceLROnPlateau with `CosineDecayRestarts`
6. **Mixup augmentation** — blend pairs of training images

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Finalize word, add to sentence |
| `BACKSPACE` | Delete last letter |
| `ENTER` | Clear current word |
| `Q` (OpenCV mode) | Quit |


## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_training_pipeline.ipynb` | Full EDA, preprocessing, training, evaluation |

Run with:
```bash
jupyter notebook notebooks/
```

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit: `git commit -m 'feat: add my feature'`
4. Push & open a PR