# A YOLOv12-ViT Hybrid-Based Multi-Feature Detection Framework with Voice Assistant for Enhanced Mobility and Independence of Visually Impaired Persons

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg?style=flat-square&logo=pytorch)](https://pytorch.org/) [![YOLOv12](https://img.shields.io/badge/YOLOv12-Latest-green.svg?style=flat-square&logo=opencv)](https://github.com/ultralytics/ultralytics) [![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE) [![Status](https://img.shields.io/badge/Status-Active-success.svg?style=flat-square)]() 
</div>

**Real-Time Multi-Modal Assistive Technology for Environmental Awareness**

[Description](#-description) • [Dataset Information](#-dataset-information) • [Code Information](#-code-information) • [Installation](#-installation--environment-setup) • [Usage Instructions](#-usage-instructions) • [Requirements](#-requirements) • [Methodology](#-methodology) • [Citations](#-citations) • [License](#-license--contribution-guidelines)

---

## Table of Contents

- [Description](#-description)
- [Dataset Information](#-dataset-information)
- [Code Information](#-code-information)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Usage Instructions](#-usage-instructions)
- [Requirements](#-requirements)
- [Methodology](#-methodology)
- [Citations](#-citations)
- [License & Contribution Guidelines](#-license--contribution-guidelines)

---

## 📋 Description

This repository provides the complete implementation of a real-time, multi-feature assistive vision framework designed for visually impaired users. The system integrates five core perception modules into a single unified pipeline with voice-based auditory feedback, enabling hands-free, context-aware navigation assistance in both indoor and outdoor environments.

The main inference system (`app.py`) combines:

- **Object Detection** – Detects navigation-critical environmental objects (vehicles, chairs, doors, persons, roads, stairs, tables, trees, walls) in real time.
- **Bangladesh Currency Recognition** – Identifies 10 denominations of Bangladeshi banknotes and coins for independent financial transactions.
- **Footpath Safety Classification** – Classifies pedestrian walkways into four safety categories: Free for Use, Fully Occupied, Not Safe for Use, and Partially Occupied.
- **Known/Unknown Face Recognition** – Recognizes enrolled individuals using ArcFace-based deep metric learning with an open-set recognition strategy.
- **Multilingual OCR (Bangla + English)** – Reads real-world text from signboards, labels, posters, and public notices using EasyOCR.

The training and evaluation system (`run.py`) supports multi-model experiments across multiple datasets, including the proposed **YOLOv12-ViT hybrid model**, which extends the YOLOv12 R-ELAN backbone with a lightweight Vision Transformer (ViT) encoder block inserted after the backbone to capture long-range global context while maintaining real-time CPU inference.

---

## 📊 Dataset Information

Three task-specific datasets were used to train and evaluate the system. All datasets are publicly available via Kaggle.

### Dataset 1: Custom Object Detection Dataset

- **Source:** Authors' own collection — Pabna University of Science and Technology (PUST), Bangladesh (indoor/outdoor campus environments)
- **Kaggle:** https://www.kaggle.com/datasets/uzzalhasan/custom-object-detection-dataset
- **Classes (9):** Vehicle, Chair, Door, Man, Road, Stair, Table, Tree, Wall
- **Images:** 703 raw → 1,829 augmented training images
- **Split:** 563 train / 71 validation / 69 test
- **Format:** YOLO bounding box `.txt` (annotated via Roboflow)

### Dataset 2: Bangladesh Currency Detection Dataset

- **Source:** Authors' own collection — currency notes/coins photographed from PUST students, teachers, and officers under varied real-world conditions
- **Kaggle:** https://www.kaggle.com/datasets/uzzalhasan/bd-currency
- **Classes (10):** 1 Tk, 2 Tk, 5 Tk, 10 Tk, 20 Tk, 50 Tk, 100 Tk, 200 Tk, 500 Tk, 1000 Tk
- **Images:** 1,627 raw → 3,801 augmented training images
- **Split:** 1,270 train / 168 validation / 189 test
- **Format:** YOLO bounding box `.txt` (annotated via Roboflow)

### Dataset 3: Footpath Detection Dataset

- **Source:** ⚠️ **Third-party public dataset** — directly borrowed from the Roboflow Universe repository (not handcrafted by the authors)
- **Kaggle:** https://www.kaggle.com/datasets/uzzalhasan/footpath-detection
- **Original Repository:** https://app.roboflow.com/md-uzzal-mia/projects
- **Classes (4):** Free for Use, Fully Occupied, Not Safe for Use, Partially Occupied
- **Images:** 4,884 (pre-split by original publisher)
- **Split:** 3,914 train / 735 validation / 235 test
- **Format:** YOLO bounding box `.txt`

> **Third-Party Dataset Acknowledgement:** The Footpath Detection Dataset is a third-party public dataset sourced from the Roboflow Universe platform. The authors acknowledge its use and direct users to https://app.roboflow.com/md-uzzal-mia/footpath-occupancy-detection-cwxe8/1 for the applicable repository license terms and access to individual dataset files being reproduced.

### Dataset Config Keys in Code (`configs/datasets.py`)

| Key                 | Description                                    |
| ------------------- | ---------------------------------------------- |
| `currency`          | Bangladesh currency detection                  |
| `footpath`          | Footpath safety detection                      |
| `blind_assistant`   | Custom object detection                        |
| `visually_impaired` | External test-only dataset (Roboflow Universe) |

---

## 💻 Code Information

- **Language:** Python 3.10+
- **Detection Framework:** Ultralytics (YOLOv12 with custom ViT block injection via forward hook in `models/builder.py`)
- **Face Recognition:** ArcFace-based deep metric learning (`face_recognition` / InsightFace); cosine similarity threshold τ = 0.50
- **OCR Engine:** EasyOCR (CNN + LSTM + CTC decoding; Bangla + English)
- **Voice Feedback:** gTTS (Bangla dynamic TTS) + pre-recorded MP3 clips via `pygame` (non-blocking, event-driven)
- **Repository:** https://github.com/uzzal2200/A-YOLOv12-ViT-Hybrid-Based-Multi-Feature-Detection-Framework-with-Voice-Assistant-for-Enhanced-Mobil

### Repository Structure

```text
Object-text-detection-for-visually-impaired/
├── app.py                          # Main real-time inference + voice assistant
├── run.py                          # Training, evaluation, multi-model experiments
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── audio/                          # Pre-recorded MP3 class audio files
├── configs/
│   ├── __init__.py
│   ├── datasets.py                 # Dataset path and YAML config definitions
│   └── models.py                   # Model key definitions
├── datasets/
│   ├── __init__.py
│   └── downloader.py               # Roboflow dataset downloader
├── models/
│   ├── __init__.py
│   ├── builder.py                  # YOLOv12-ViT hybrid builder (ViT forward-hook injection)
│   └── vit_block.py                # Lightweight ViT encoder block (MHSA + FFN + residual)
├── trainers/
│   ├── __init__.py
│   ├── trainer.py                  # Training loop wrapper
│   └── tester.py                   # Evaluation and metrics computation
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── visualizer.py
├── Known_unknown_detection/
│   ├── known_unknown_detection.py  # Standalone ArcFace face recognition
│   ├── evaluation_metrices.py      # FAR / FRR / Accuracy computation
│   ├── known_faces_folder/         # Place known-person images here
│   └── movement_logs/              # Timestamped recognition event logs
├── OCR detection/
│   ├── OCR_Bangla_english.py       # Standalone Bangla + English OCR module
│   └── evaluation_metrices.py      # CER / WER / WRA computation
├── Notebook Experiment/
│   ├── Bangladeshi_Currency_detection_with_yolov12n_pt.ipynb
│   ├── custom_object_detection_with_yolov12n_pt.ipynb
│   └── Footpath_detection_yolov12n_pt.ipynb
└── Save Model/
    ├── Bangldesh currencey Detection/best.pt
    ├── Footpath Detection/best.pt
    └── Object detection Custom dataset/best.pt
```

### Model Keys (`configs/models.py`)

| Key           | Description                               |
| ------------- | ----------------------------------------- |
| `yolov5n`     | YOLOv5 nano — comparison baseline         |
| `yolov8n`     | YOLOv8 nano — comparison baseline         |
| `yolov11n`    | YOLOv11 nano — comparison baseline        |
| `yolov12n`    | YOLOv12 nano (no ViT) — ablation baseline |
| `yolov12_vit` | **Proposed: YOLOv12-ViT hybrid**          |

### Runtime Controls (during `app.py`)

| Key | Action                         |
| --- | ------------------------------ |
| `C` | Toggle currency module         |
| `F` | Toggle footpath module         |
| `O` | Toggle object module           |
| `X` | Toggle face recognition module |
| `R` | Toggle OCR module              |
| `Q` | Quit                           |

---

## 🔧 Installation & Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/uzzal2200/YOLOv12-Based-Multi-Feature-Detection-Framework-with-Voice-Assistant.git
cd YOLOv12-Based-Multi-Feature-Detection-Framework-with-Voice-Assistant
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install System Dependencies

**Linux:**

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr ffmpeg
```

**Windows:**

- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- FFmpeg: https://ffmpeg.org/download.html (add `ffmpeg/bin` to PATH)
- For `dlib`: Visual C++ Build Tools required

**macOS:**

```bash
brew install tesseract ffmpeg
```

### 4. Optional: Set Tesseract Path (Windows)

If Tesseract is not on PATH, set in `app.py`:

```python
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

### 5. Obtain Model Weights

Pre-trained weights must be placed at these exact paths (required by `app.py`):

```
Save Model/Bangldesh currencey Detection/best.pt
Save Model/Footpath Detection/best.pt
Save Model/Object detection Custom dataset/best.pt
```

Train your own with `run.py` or download from the GitHub Releases page.

---

## 🚀 Usage Instructions

### How to Load Datasets

Download datasets directly from Kaggle:

```bash
kaggle datasets download -d uzzalhasan/custom-object-detection-dataset
kaggle datasets download -d uzzalhasan/bd-currency
kaggle datasets download -d uzzalhasan/footpath-detection
```

Or use the built-in Roboflow downloader via `run.py` (requires a Roboflow API key in `configs/datasets.py`).

### Run Real-Time Inference (Webcam)

```bash
python app.py
```

### Run Real-Time Inference (Video File)

```bash
python app.py --video path/to/video.mp4
```

### Training and Evaluation

```bash
# Run all configured datasets and models
python run.py

# Run a specific dataset only
python run.py --datasets currency

# Run a specific model only
python run.py --models yolov12_vit

# Override training hyperparameters
python run.py --epochs 50 --batch 8 --imgsz 640

# Skip Roboflow download (use local YAML paths)
python run.py --skip-download

# Test only with custom weights
python run.py --datasets visually_impaired --test-weights path/to/best.pt
```

### Standalone OCR Module

```bash
python "OCR detection/OCR_Bangla_english.py"
```

### Standalone Face Recognition Module

```bash
python Known_unknown_detection/known_unknown_detection.py
```

### Face Database Setup

Add known-person images to `Known_unknown_detection/known_faces_folder/`. Use clear, frontal images with the person's name as the filename (used as the identity label).

---

## 📦 Requirements

```
Python >= 3.10
torch >= 2.0.0
torchvision >= 0.15.0
ultralytics >= 8.0.0
easyocr >= 1.7.0
opencv-python >= 4.7.0
numpy >= 1.23.0
pillow >= 9.4.0
pygame >= 2.3.0
gTTS >= 2.3.1
pydub >= 0.25.1
pytesseract >= 0.3.10
face-recognition >= 1.3.0
dlib >= 19.24.0
insightface >= 0.7.3
onnxruntime >= 1.14.0
scikit-learn >= 1.2.0
roboflow >= 1.1.0
```

Install all requirements:

```bash
pip install -r requirements.txt
```

> **CPU Inference:** All benchmarks were conducted on CPU only. The system achieves 6–10 FPS at 95–150 ms/frame. Model size: 5.27 MB. No GPU required.

> **Bangla TTS:** `gTTS` requires an active internet connection.

---

## 🔬 Methodology

This section describes the complete data processing, model design, training, and evaluation methodology used in this work.

### 1. Data Collection

| Dataset            | Collection Method                                                  | Environment                  |
| ------------------ | ------------------------------------------------------------------ | ---------------------------- |
| Object Detection   | Authors photographed campus scenes at PUST using DSLR              | Indoor + outdoor             |
| Currency Detection | Mobile phone photographs of banknotes/coins from PUST participants | Indoor (varied backgrounds)  |
| Footpath Detection | Sourced from Roboflow Universe (third-party)                       | Real-world pedestrian scenes |

### 2. Image Annotation

Object detection and currency images were manually annotated in YOLO bounding box format using Roboflow (https://roboflow.com). Each annotation encodes `[class_id, x_center, y_center, width, height]` normalized to image dimensions. Consistency across all classes was enforced using Roboflow's audit tools; hand-checking removed label errors.

### 3. Preprocessing

Applied to all images via Roboflow prior to training:

- Orientation correction (EXIF metadata)
- Contrast enhancement (low-light normalization)
- Color and exposure normalization
- Systematic file renaming for reproducibility

### 4. Data Augmentation

Applied to training sets only:

| Augmentation         | Parameters             |
| -------------------- | ---------------------- |
| Horizontal flip      | Mirror image           |
| Random rotation      | −10° to +10°           |
| Brightness variation | ±15%                   |
| Gaussian motion blur | Kernel-based           |
| Pixel noise          | 0.1% random corruption |

### 5. Model Architecture — YOLOv12-ViT Hybrid

The YOLOv12-ViT model is constructed as follows:

1. **Backbone (R-ELAN):** Residual Efficient Layer Aggregation Network with 7×7 separable convolutions + FlashAttention for efficient multi-scale local feature extraction.
2. **ViT Encoder Block (proposed):** Inserted immediately after the backbone via a forward hook. The feature map F ∈ R^(C×H×W) is flattened to token sequence X ∈ R^(HW×C), processed by Multi-Head Self-Attention (MHSA) and a Feed-Forward Network (FFN) with residual connection, then reshaped back to spatial dimensions. This captures long-range dependencies without modifying the neck or detection head.
3. **Neck:** Area-based attention with FlashAttention for multi-scale feature aggregation.
4. **Detection Head:** Multi-scale branches with combined bounding-box regression and classification loss.

### 6. Training Configuration

| Parameter             | Value                  |
| --------------------- | ---------------------- |
| Epochs                | 100                    |
| Batch size            | 16                     |
| Input resolution      | 640 × 640              |
| Optimizer             | SGD (momentum = 0.937) |
| Initial learning rate | 0.01                   |
| Parameters            | ~2.57 M                |
| Inference GFLOPs      | 6.5                    |
| Framework             | Ultralytics            |

### 7. Face Recognition

- ArcFace loss for discriminative face embedding training
- Cosine similarity matching against an offline enrolled-faces database
- Open-set decision: Known if similarity ≥ τ = 0.50, else Unknown
- FAR = 0.00%, FRR = 3.03% at τ = 0.50 (empirically validated)
- All processing is performed locally; no biometric data transmitted to cloud

### 8. OCR

- EasyOCR framework: CNN feature extractor → LSTM sequence model → CTC decoder
- Languages: Bangla + English
- Bengali-compatible TrueType fonts for accurate Unicode glyph display
- Runs in parallel with all detection modules during real-time inference

### 9. Evaluation Method

The model is evaluated on the held-out test splits (never used during training or validation). **Comparative Performance Analysis** compares YOLOv12-ViT against four baselines — YOLOv5n, YOLOv8n, YOLOv11n, and YOLOv12n (without ViT) — each trained from scratch with identical hyperparameters and evaluated on the same test sets. Generalization is further assessed on an unseen external dataset (Roboflow Universe Visually Impaired Dataset, 30 classes) without fine-tuning.

### 10. Assessment Metrics

All metrics are justified for their role in assistive navigation:

**Detection Tasks:**

| Metric       | Formula                     | Rationale                                                 |
| ------------ | --------------------------- | --------------------------------------------------------- |
| Precision    | TP/(TP+FP)                  | Minimizes false alarms; critical for user trust           |
| Recall       | TP/(TP+FN)                  | Minimizes missed detections; critical for safety          |
| mAP@0.5      | Mean AP at IoU ≥ 0.5        | Standard combined localization + classification benchmark |
| mAP@0.5–0.95 | Mean AP across IoU 0.5–0.95 | Stricter spatial precision assessment                     |

**Face Recognition:**

| Metric               | Formula               | Rationale                                           |
| -------------------- | --------------------- | --------------------------------------------------- |
| Recognition Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall identification correctness                  |
| Recall               | TP/(TP+FN)            | Sensitivity to known individuals                    |
| FAR                  | FP/(FP+TN)            | Security: prevents unauthorized identity acceptance |
| FRR                  | FN/(TP+FN)            | Usability: avoids repeated rejection frustration    |

**OCR:**

| Metric | Formula                          | Rationale                                                 |
| ------ | -------------------------------- | --------------------------------------------------------- |
| CER    | Edit Distance / Total Characters | Character-level accuracy for complex multilingual scripts |
| WER    | Edit Distance / Total Words      | Word-level accuracy                                       |
| WRA    | (1−WER)×100                      | Human-interpretable readability score                     |

---

## 📚 Citations

If you use this code, models, or datasets in your research, please cite:

```bibtex
@article{mia2025yolov12vit,
  title={A YOLOv12-ViT Hybrid-Based Multi-Feature Detection Framework with Voice Assistant
         for Enhanced Mobility and Independence of Visually Impaired Persons},
  author={Mia, Md. Uzzal and Debnath, Sajib and Islam Mridul, Md Shafiqul and
          Mulk, Md. Taz Warul and Hosain, Md. Sarwar and Shimamura, Tetsuya},
  journal={PeerJ Computer Science},
  year={2025},
  note={Under Review}
}
```

**Key references used in this work:**

- Tian, Y. et al. (2025). YOLOv12. _arXiv preprint_.
- Dosovitskiy, A. (2020). An Image is Worth 16×16 Words. _arXiv preprint_.
- Deng, J. et al. (2022). ArcFace. _IEEE TPAMI_, 44(10):5962–5979.
- Vaswani, A. et al. (2017). Attention is All You Need. _arXiv preprint_.
- Redmon, J. et al. (2016). You Only Look Once. _CVPR_, pp. 779–788.
- Roboflow Universe. Footpath Detection Dataset. https://app.roboflow.com/md-uzzal-mia/footpath-occupancy-detection-cwxe8/1

---

## 📄 License & Contribution Guidelines

### License

This project's source code and authors' own datasets are released under the **MIT License**. See [LICENSE](LICENSE) for complete terms.

> ⚠️ **Third-Party Dataset:** The Footpath Detection Dataset is sourced from the Roboflow Universe platform and subject to its original publisher's license. Refer to https://app.roboflow.com/md-uzzal-mia/footpath-occupancy-detection-cwxe8/1 for license terms before use or redistribution.

### Contribution Guidelines

Contributions, bug reports, and feature requests are welcome.

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with clear messages.
4. Submit a pull request with a description of your changes.

All contributions should be compatible with the MIT License.

---

## ⚠️ Notes

- Keep model folder names unchanged, including current spelling (`Bangldesh currencey Detection`) to avoid path errors.
- Audio files in `audio/` must retain exact filenames (explicit label-to-file mapping).
- Bangla TTS (`gTTS`) requires an internet connection.
- On Windows, `face_recognition` requires `dlib` build support.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO framework
- [OpenCV](https://opencv.org/) — Computer vision
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Multilingual OCR
- [gTTS](https://github.com/pndurette/gTTS) — Text-to-Speech
- [InsightFace](https://github.com/deepinsight/insightface) — ArcFace
- [Roboflow](https://roboflow.com/) — Annotation platform and dataset hosting

---

_Last Updated: May 2, 2026_
