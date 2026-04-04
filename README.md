# A YOLOv12-ViT Hybrid-Based Multi-Feature Detection Framework with Voice Assistant for Enhanced Mobility and Independence of Visually Impaired Persons

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![YOLOv12](https://img.shields.io/badge/YOLOv12-Latest-green.svg?style=flat-square&logo=opencv)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg?style=flat-square)]()

**Real-Time Multi-Modal Assistive Technology for Environmental Awareness**

[Features](#-key-features) • [Quick Start](#-quick-start) • [Installation](#-installation--environment-setup) • [Datasets](#-datasets) • [Project Structure](#-project-structure)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Datasets](#-datasets)
- [Models](#-models)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Quick Start](#-quick-start)
- [Training and Evaluation](#-training-and-evaluation)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## Overview

This repository provides a real-time assistive framework for visually impaired users using multiple computer vision modules with voice feedback.

The main inference system (`app.py`) combines:

- Object detection
- Bangladeshi currency detection
- Footpath safety detection
- Known/unknown face recognition
- OCR with Bangla speech output

The training/evaluation system (`run.py`) supports multi-model experiments across multiple datasets, including a YOLOv12 + ViT hybrid model.

## Key Features

### Detection Capabilities

| Module | Detection Classes | Audio Output | Use Case |
|--------|------------------|--------------|----------|
| Object Detection | Vehicle, Chair, Door, Man, Road, Stair, Table, Tree, wall (9 classes) | Pre-recorded MP3 | Environmental awareness |
| Currency Detection | 1 Tk - 1000 Tk denominations (9 classes in current mapping) | Pre-recorded MP3 | Financial independence |
| Footpath Safety | free for use, Fully Occupied, Not for Safe, Partially Occupied (4 classes) | Pre-recorded MP3 | Safe navigation |
| Face Recognition | known_face / unknown_face | Pre-recorded MP3 | Social interaction |
| OCR Detection | English text extraction in app; Bangla-English OCR in standalone module | Dynamic Bangla gTTS + console logs | Information access |

### Runtime Interaction

During live inference (`app.py`):

- `C`: Toggle currency module
- `F`: Toggle footpath module
- `O`: Toggle object module
- `X`: Toggle face module
- `R`: Toggle OCR module
- `Q`: Quit

## Datasets

### Dataset 1: Custom Object Detection
- Source (documented): Kaggle custom object dataset
- Classes: Vehicle, Chair, Door, Man, Road, Stair, Table, Tree, Wall
- Application: General environmental object detection

### Dataset 2: Bangladeshi Currency Detection
- Source (documented): Kaggle BD currency dataset
- Classes: Bangladeshi denominations
- Application: Currency recognition

### Dataset 3: Footpath Detection
- Source (documented): Kaggle footpath dataset
- Classes: Free/Occupied/Unsafe/Partially occupied path conditions
- Application: Navigation safety

### Dataset 4: Face Recognition Database
- Storage: `Known_unknown_detection/known_faces_folder/`
- Format: JPG/PNG
- Application: Known/unknown person identification

### Dataset Config Keys in Code (`configs/datasets.py`)

- `currency`
- `footpath`
- `blind_assistant`
- `visually_impaired` (test-only dataset)

## Models

### Training Model Keys (`configs/models.py`)

- `yolov5n`
- `yolov8n`
- `yolov11n`
- `yolov12n`
- `yolov12_vit`

### YOLOv12-ViT Hybrid

The hybrid model uses YOLOv12n base weights and injects a ViT block through a forward hook at a backbone layer (`models/builder.py`).

## Installation & Environment Setup

### 1) Python Dependencies

```bash
pip install -r requirements.txt
```

### 2) System Dependencies

Required for full functionality:

- Tesseract OCR (for `pytesseract`)
- FFmpeg (for `pydub` audio handling)

Windows:

- Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
- Install FFmpeg and add `ffmpeg/bin` to system `PATH`
- If needed for face recognition builds: Visual C++ Build Tools

Linux:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr ffmpeg
```

macOS:

```bash
brew install tesseract ffmpeg
```

### 3) Optional Tesseract Manual Path (Windows)

If Tesseract is not in PATH, set it in `app.py`:

```python
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

### 4) Required Inference Weights

`app.py` loads these exact paths:

- `Save Model/Bangldesh currencey Detection/best.pt`
- `Save Model/Footpath Detection/best.pt`
- `Save Model/Object detection Custom dataset/best.pt`

## Quick Start

### Real-time Inference (Webcam)

```bash
python app.py
```

### Real-time Inference (Video)

```bash
python app.py --video path/to/video.mp4
```

### Standalone OCR Module

```bash
python "OCR detection/OCR_Bangla_english.py"
```

### Standalone Face Module

```bash
python Known_unknown_detection/known_unknown_detection.py
```

## Training and Evaluation

Main entry point: `run.py`

```bash
# Run all configured datasets/models
python run.py

# Run only specific dataset(s)
python run.py --datasets currency

# Run only specific model(s)
python run.py --models yolov12_vit

# Override training hyperparameters
python run.py --epochs 50 --batch 8 --imgsz 640

# Skip Roboflow downloading and use local YAML paths
python run.py --skip-download

# Test-only run with custom weights
python run.py --datasets visually_impaired --test-weights path/to/best.pt
```

## Audio Assets

All class audio files are stored in `audio/` and must keep exact filenames because the app uses explicit label-to-file mapping.

## Face Database Setup

Add known people images to:

- `Known_unknown_detection/known_faces_folder/`

Recommendations:

- Use clear frontal images
- One prominent face per image
- Keep filename as person identity label

## Project Structure

```text
Object-text-detection-for-visually-impaired/
├── app.py
├── run.py
├── requirements.txt
├── README.md
├── LICENSE
├── audio/
├── configs/
│   ├── __init__.py
│   ├── datasets.py
│   └── models.py
├── datasets/
│   ├── __init__.py
│   └── downloader.py
├── models/
│   ├── __init__.py
│   ├── builder.py
│   └── vit_block.py
├── trainers/
│   ├── __init__.py
│   ├── trainer.py
│   └── tester.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── visualizer.py
├── Known_unknown_detection/
│   ├── known_unknown_detection.py
│   ├── evaluation_metrices.py
│   ├── known_faces_folder/
│   └── movement_logs/
├── OCR detection/
│   ├── OCR_Bangla_english.py
│   └── evaluation_metrices.py
├── Notebook Experiment/
│   ├── Bangladeshi_Currency_detection_with_yolov12n_pt.ipynb
│   ├── custom_object_detection_with_yolov12n_pt.ipynb
│   └── Footpath_detection_yolov12n_pt.ipynb
└── Save Model/
    ├── Bangldesh currencey Detection/
    │   └── best.pt
    ├── Footpath Detection/
    │   └── best.pt
    └── Object detection Custom dataset/
        └── best.pt
```

## Notes

- Bangla TTS (`gTTS`) requires internet connection.
- On Windows, `face_recognition` installation may require `dlib` build support.
- Keep model folder names unchanged, including current spelling in `Bangldesh currencey Detection`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- Ultralytics
- OpenCV
- dlib / face_recognition
- Tesseract OCR
- EasyOCR
- gTTS

Last Updated: April 4, 2026
