import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def split_dataset(raw_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset menjadi train, validation, dan test
    
    Parameters:
    - raw_path: Path ke folder raw dataset
    - output_path: Path ke folder output (processed)
    - train_ratio: Proporsi data training (default: 0.7)
    - val_ratio: Proporsi data validation (default: 0.15)
    - test_ratio: Proporsi data testing (default: 0.15)
    - seed: Random seed untuk reproducibility
    """
    random.seed(seed)
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    
    print("=" * 70)
    print("DATA SPLITTING")
    print("=" * 70)
    print(f"Train ratio: {train_ratio*100:.0f}%")
    print(f"Validation ratio: {val_ratio*100:.0f}%")
    print(f"Test ratio: {test_ratio*100:.0f}%")
    print(f"Random seed: {seed}")
    print("=" * 70)
    
    for class_name in ["matoa", "namnam", "kupa"]:
        print(f"\n📂 Processing: {class_name}")
        print("-" * 70)
        
        class_path = raw_path / class_name
        
        if not class_path.exists():
            print(f"⚠️  Folder {class_name} tidak ditemukan! Skip...")
            continue
        
        # Ambil semua gambar
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        print(f"Total images: {len(images)}")
        
        # Split train dan temp (val + test)
        train_imgs, temp_imgs = train_test_split(
            images,
            test_size=(val_ratio + test_ratio),
            random_state=seed
        )
        
        # Split temp menjadi val dan test
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed
        )
        
        print(f"  - Train: {len(train_imgs)} images")
        print(f"  - Validation: {len(val_imgs)} images")
        print(f"  - Test: {len(test_imgs)} images")
        
        # Copy files ke folder tujuan
        for split_name, split_imgs in [
            ("train", train_imgs),
            ("validation", val_imgs),
            ("test", test_imgs)
        ]:
            dest_folder = output_path / split_name / class_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_imgs:
                dest_path = dest_folder / img_path.name
                shutil.copy2(str(img_path), str(dest_path))
        
        print(f"✓ {class_name} selesai!")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Hitung total per split
    for split_name in ["train", "validation", "test"]:
        split_path = output_path / split_name
        total = 0
        for class_name in ["matoa", "namnam", "kupa"]:
            class_path = split_path / class_name
            if class_path.exists():
                count = len(list(class_path.glob("*")))
                total += count
                print(f"{split_name}/{class_name}: {count} images")
        print(f"  TOTAL {split_name}: {total} images")
        print()
    
    print("=" * 70)
    print("✓ Data splitting selesai!")

# ============================================================================
# JALANKAN SCRIPT
# ============================================================================
if __name__ == "__main__":
    RAW_PATH = "../1_dataset/raw"
    OUTPUT_PATH = "../1_dataset/processed"
    
    print("Starting data splitting...")
    print(f"Raw path: {RAW_PATH}")
    print(f"Output path: {OUTPUT_PATH}\n")
    
    split_dataset(RAW_PATH, OUTPUT_PATH)