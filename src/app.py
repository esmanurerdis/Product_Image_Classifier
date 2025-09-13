import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from src.infer import predict_image  

# Görselleri modelin istediği boyuta getir
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify(image):
    image = transform(image).unsqueeze(0)  # Batch boyutu ekle
    label, proba = predict_image(image)    # predict_image: modeli load edip tahmin döndüren fonksiyon
    return f"Prediction: {label} ({proba:.2f})"

# Arayüz
iface = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Mini CV Project - Image Classifier",
    description="Upload a product image and get its classification.",
)

if __name__ == "__main__":
    iface.launch()
