import torch
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import json
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = Path("data/processed")   # Test set klasörü
MODEL_PATH = Path("models/best_model.pth")
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sınıf isimlerini al
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

# Test veri seti ve loader
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
test_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Modeli yükle
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix ve classification report
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

# JSON olarak kaydet
with open(REPORTS / "metrics.json", "w") as f:
    json.dump(report, f, indent=2)

# Confusion matrix görselleştirme
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORTS / "confusion_matrix.png", dpi=150)
plt.show()

print("✅ Evaluation completed. Reports saved in 'reports/'")
