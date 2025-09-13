# 🚀Product Image Classifier

• A computer vision project that classifies product images using MobileNetV2.
• The project includes model fine-tuning, evaluation, and a GUI for easy user interaction.
• The GUI is implemented with both Tkinter and Gradio, and the final choice is based on usability comparison.

---
 
# 🖼️ Project Overview
![GUI Screenshot](reports/Product_Image_Classifier.png)

---

# Features⭐

• Model: MobileNetV2 fine-tuned on custom image classes
• GUI: Gradio interface for image upload and classification; Tkinter used for comparison
• Evaluation: Confusion matrix generated for small dataset
• Fine-tuning: Pretrained weights adapted to custom classes
/
• 🐍 **Python Scripts**: `app.py`, `train.py`, `download_data.py`  
• 📊 **Data Visualization**: `matplotlib` & `seaborn`  
• 🧠 **Machine Learning**: Fine-tuned MobileNetV2 for image classification  
• 🖼️ **Image Processing**: `PIL` for resizing & preprocessing  
• 💻 **GUI Interface**: Gradio / Tkinter integration  
• 📁 **Data & Reports**: Organized dataset, trained models, confusion matrix, and metrics

---

# Project Structure📁

```
Product_Image_Classifier/
├── data/
│   └── processed/
│       ├── class1/
│       │   ├── 0.png
│       │   ├── 1.png
│       │   └── ...
│       └── class2/
│           ├── 0.png
│           ├── 1.png
│           └── ...
├── models/
│   └── best_model.pth
├── reports/
│   ├── Product_Image_Classifier.png
│   ├── confusion_matrix.png
│   └── metrics.json
├── src/
│   ├── app.py            # Gradio arayüzü
│   ├── download_data.py  # Mini veri seti indirme scripti
│   ├── train.py          # Model eğitimi scripti
│   ├── evaluate.py       # Model değerlendirme scripti
│   ├── infer.py          # Tek görsel sınıflandırma
│   └── preprocess.py     # Veri ön işleme
├── .gitignore
├── README.md
└── requirements.txt

```
---


# Quick Start🏁

## Install requirements📦
```
pip install -r requirements.txt
```

## Run Gradio interface🌐
```
python src/infer.py
```
---

# Fine-tuning Explanation📝

We used a pretrained MobileNetV2 and only trained the classifier head on our small dataset.
This approach:

• Speeds up training
• Reduces overfitting on small datasets
• Maintains strong feature extraction from pretrained convolutional layers

---


# Technology Stack🛠️

• Python 3.9+ 🐍 

• PyTorch & Torchvision🔥

• Gradio / Tkinter🖥️

• NumPy, Pandas📊

• Matplotlib, Seaborn📈

---


# 🇹🇷 Türkçe Özet

• Bu proje, MobileNetV2 ile ürün görsellerini sınıflandıran bir uygulamadır.

• Model Eğitimi & Fine-tuning: Önceden eğitilmiş MobileNetV2 modeli kendi sınıflarımıza göre adapte edildi.

• Arayüz Karşılaştırması: Tkinter ve Gradio arayüzleri test edildi; kullanıcı deneyimi ve kullanım kolaylığı göz önünde bulundurularak Gradio tercih edildi.

• Değerlendirme: Mini veri seti üzerinde sınıflandırma sonuçları confusion matrix ile görselleştirildi.
