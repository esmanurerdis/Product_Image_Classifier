import gradio as gr
from PIL import Image
import torch
from torchvision import transforms, models
from pathlib import Path

MODEL_PATH = Path("models/best_model.pth")
DATA_DIR = Path("data/processed")  # burada her klasör bir sınıf adı olacak: data/processed/class1, class2...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# classes listesini oluştur
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

# modeli yükle
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

def predict_image(img: Image.Image):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()
    return classes[pred]

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Product Image Classifier",
    description="Upload a product image and get its classification."
)

if __name__ == "__main__":
    iface.launch()

