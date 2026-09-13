# 🍎 Fruit Quality & Adulteration Classifier (CNN)

A deep learning powered web app that classifies fruit images as **Fresh**, **Rotten**, or **Formalin-mixed** (chemically adulterated) using a Convolutional Neural Network trained with PyTorch, served through a FastAPI backend.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Uvicorn-2A2A2A?style=for-the-badge&logo=gunicorn&logoColor=white" />
  <img src="https://img.shields.io/badge/Jinja2-B41717?style=for-the-badge&logo=jinja&logoColor=white" />
  <img src="https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

<p align="left">
  <img src="https://img.shields.io/github/license/zakir-maswani/Fruits-Quality-Classification-CNN?style=flat-square" />
  <img src="https://img.shields.io/github/last-commit/zakir-maswani/Fruits-Quality-Classification-CNN?style=flat-square" />
  <img src="https://img.shields.io/github/stars/zakir-maswani/Fruits-Quality-Classification-CNN?style=flat-square" />
</p>

---

## 🎥 Demo

<!-- Replace the line below with your demo video (upload the video/gif to the repo or use a GitHub-hosted link) -->
[![Demo Video](assets/demo-thumbnail.png)](assets/demo-video.mp4)

> 📌 Add your demo video/gif here — drag & drop it into this section on GitHub, or link a YouTube/Loom video.

---

## 📖 Overview

Fruit adulteration (e.g. injecting formalin to artificially extend shelf life) and spoilage are common food-safety concerns. This project trains a CNN on images of five fruits (**Apple, Banana, Grape, Mango, Orange**) organized by condition, and classifies any fruit image into one of three quality categories:

- ✅ **Fresh**
- 🦠 **Rotten**
- ⚠️ **Formalin-mixed**

The trained model is deployed behind a simple FastAPI web app where users can upload an image and instantly get a prediction with a confidence score.

---

## ✨ Features

- CNN trained from scratch with PyTorch (Conv2D + MaxPooling + Fully Connected layers)
- Image preprocessing pipeline (resize, normalize, tensor conversion)
- FastAPI backend serving a REST `/predict` endpoint
- Simple HTML front-end (Jinja2 templates) to upload and preview images
- Returns predicted class + confidence percentage

---

## 🗂️ Project Structure

```
.
├── data_preprocessing_and_model_training.ipynb   # Data loading, CNN architecture, training loop
├── main.py                                       # FastAPI app serving the trained model
├── fruit_quality_and_adulteration_classifier.pth # Trained model weights (not committed if large)
├── templates/
│   └── index.html                                # Upload UI
├── static/                                        # CSS/JS/assets for the front-end
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

```
Conv2D(3 → 32) → ReLU → MaxPool
Conv2D(32 → 64) → ReLU → MaxPool
Conv2D(64 → 128) → ReLU → MaxPool
Flatten
Linear(16*16*128 → 256) → ReLU
Linear(256 → 3)   # Formalin-mixed, Fresh, Rotten
```

- **Input size:** 128x128 RGB images
- **Loss function:** CrossEntropyLoss
- **Optimizer:** Adam (lr = 0.001)

---

## 📊 Dataset

Images are organized by fruit type, and within each fruit, by condition:

```
Dataset/
├── train/
│   ├── Apple/
│   │   ├── Formalin-mixed/
│   │   ├── Fresh/
│   │   └── Rotten/
│   ├── Banana/
│   ├── Grape/
│   ├── Mango/
│   └── Orange/
├── test/
└── valid/
```

A custom PyTorch `Dataset` class walks each fruit folder and labels every image based on its **condition sub-folder** (Formalin-mixed / Fresh / Rotten), ignoring the fruit type — since the goal is quality classification, not fruit classification.

> Dataset source: *(add link/credit here, e.g. Kaggle dataset URL, if applicable)*

---

## ⚙️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Place your trained model weights (`fruit_quality_and_adulteration_classifier.pth`) in the project root.

---

## 🚀 Usage

### Run the web app

```bash
python main.py
```

Then open your browser at:
```
http://127.0.0.1:8000
```

Upload a fruit image and get an instant prediction.

### Use the API directly

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/path/to/your/image.jpg"
```

**Example response:**
```json
{
  "prediction": "Fresh",
  "confidence": 96.42
}
```

### Train the model yourself

Open `data_preprocessing_and_model_training.ipynb` in Jupyter and run all cells — it covers data loading, the CNN architecture, training loop, and saving the model weights.

---

## 🛣️ Future Improvements

- [ ] Add data augmentation to improve generalization
- [ ] Train on a larger, more diverse dataset
- [ ] Add per-fruit + per-condition breakdown (e.g. "Apple - Rotten")
- [ ] Deploy to a cloud platform (Render/Railway/HuggingFace Spaces)
- [ ] Add model evaluation metrics (confusion matrix, precision/recall) to the notebook

---

## 🧰 Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/-Torchvision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/-Uvicorn-2A2A2A?style=flat-square&logo=gunicorn&logoColor=white" />
  <img src="https://img.shields.io/badge/-Jinja2-B41717?style=flat-square&logo=jinja&logoColor=white" />
  <img src="https://img.shields.io/badge/-Pillow-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/-HTML5-E34F26?style=flat-square&logo=html5&logoColor=white" />
  <img src="https://img.shields.io/badge/-CSS3-1572B6?style=flat-square&logo=css3&logoColor=white" />
</p>

---

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it.

---

## 🙋 Author

Built by *(your name)* — feel free to connect or raise an issue if you find a bug!
