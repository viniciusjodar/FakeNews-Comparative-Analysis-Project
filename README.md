# FakeNews Comparative Analysis Project

This repository contains experiments with **fine-tuning Transformer models** using the Hugging Face `transformers` library. The project includes evaluation before and after fine-tuning, as well as performance comparison.  
The training was performed on a **GPU (CUDA enabled)** for faster computation.

---

## 📌 Features
- Load and preprocess dataset (train/validation).
- Tokenization and dataset preparation.
- Fine-tuning Transformer models.
- Evaluation before and after training.
- Accuracy comparison.

---

## 🔧 Prerequisites
Before running the notebook, make sure you have:
- Python **3.12+**
- A machine with **NVIDIA GPU** and **CUDA drivers installed**.
- [PyTorch with CUDA support](https://pytorch.org/get-started/locally/).
- [Hugging Face Transformers](https://huggingface.co/transformers/).
- [Datasets](https://huggingface.co/docs/datasets/).
- [Accelerate](https://huggingface.co/docs/accelerate/index).

---

## ⚙️ Installation
Clone the repository and create a virtual environment:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create a virtual environment
python3 -m venv .nlp-env
source .nlp-env/bin/activate   # On Windows: .nlp-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate evaluate scikit-learn
```

---

## ▶️ Usage
Open the Jupyter Notebook and run the cells step by step:

```bash
jupyter notebook Main.ipynb
```

The notebook includes:
1. Evaluation **before fine-tuning**.
2. Fine-tuning loop with `Trainer`.
3. Evaluation **after fine-tuning**.
4. Comparison of results.

---

## 📊 Example Output
```
Evaluating the model BEFORE fine-tuning
Accuracy BEFORE Fine-Tuning: 0.65

Evaluating the model AFTER fine-tuning
Accuracy AFTER Fine-Tuning:  0.80
```

---

## 🖥️ GPU Information
This project was trained on:
- **NVIDIA GeForce RTX 3050 Laptop GPU**  
- **CUDA 12.8**  
- **Torch 2.8.0+cu128**  
- **Transformers 4.56.0**

---

## 📜 License
This project is released under the MIT License.
