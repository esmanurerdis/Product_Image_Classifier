import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path

# Klasör oluştur
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

# Transform (boyutlandırma)
transform = transforms.Compose([transforms.Resize((224,224))])

# CIFAR-10 veri setini indir (train set)
train_dataset = datasets.CIFAR10(root="data", train=True, download=True)

# Kaç görsel alacağımızı belirleme
NUM_PER_CLASS = 5
class_counts = {0:0, 1:0}  

for idx, (img, label) in enumerate(train_dataset):
    if label > 1:  
        continue
    
    if class_counts[label] >= NUM_PER_CLASS:
        continue

    class_dir = PROCESSED_DIR / train_dataset.classes[label]
    class_dir.mkdir(exist_ok=True)
    
    img.save(class_dir / f"{class_counts[label]}.png")
    class_counts[label] += 1

    if all(count >= NUM_PER_CLASS for count in class_counts.values()):
        break

print("✅ Mini dataset ready: 5 images per class downloaded.")
