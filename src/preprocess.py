import os
from pathlib import Path
from PIL import Image
import argparse
from sklearn.model_selection import train_test_split

INPUT = Path("data/raw")
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

SIZE = (224, 224)

def preprocess_images(input_dir, output_dir):
    images, labels = [], []
    for label in os.listdir(input_dir):
        label_dir = input_dir / label
        if not label_dir.is_dir():
            continue
        for img_file in os.listdir(label_dir):
            img_path = label_dir / img_file
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(SIZE)
                save_dir = output_dir / label
                save_dir.mkdir(parents=True, exist_ok=True)
                img.save(save_dir / img_file)
                images.append(str(save_dir / img_file))
                labels.append(label)
            except Exception as e:
                print(f"Skipping {img_file}: {e}")
    return images, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    preprocess_images(Path(args.input), Path(args.output))
    print("✅ Preprocessing done!")
