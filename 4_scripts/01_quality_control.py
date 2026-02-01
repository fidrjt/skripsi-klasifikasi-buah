import cv2
import os
from pathlib import Path
import shutil

def check_image_quality(image_path):
    """
    Cek kualitas gambar
    Returns: (is_valid, reason)
    """
    try:
        img = cv2.imread(str(image_path))
        
        # Check 1: File corrupted?
        if img is None:
            return False, "File corrupted"
        
        # Check 2: Resolution too low?
        height, width = img.shape[:2]
        if width < 200 or height < 200:
            return False, f"Resolution too low: {width}x{height}"
        
        # Check 3: Too blurry?
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 20:
            return False, f"Too blurry: {laplacian_var:.2f}"
        
        # Check 4: Extreme brightness?
        mean_brightness = gray.mean()
        if mean_brightness < 30:
            return False, f"Too dark: {mean_brightness:.2f}"
        if mean_brightness > 225:
            return False, f"Too bright: {mean_brightness:.2f}"
        
        return True, "OK"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

def quality_control_dataset(raw_path):
    """
    Jalankan quality control untuk seluruh dataset
    """
    raw_path = Path(raw_path)
    
    # Buat folder untuk rejected images
    rejected_path = raw_path.parent / "rejected"
    rejected_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("QUALITY CONTROL DATASET")
    print("=" * 70)
    
    total_images = 0
    total_rejected = 0
    
    # Loop untuk setiap kelas
    for class_folder in ["matoa", "namnam", "kupa"]:
        class_path = raw_path / class_folder
        
        if not class_path.exists():
            print(f"\n⚠️  Folder {class_folder} tidak ditemukan!")
            continue
        
        print(f"\n📂 Checking: {class_folder}")
        print("-" * 70)
        
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        class_rejected = 0
        
        for img_path in images:
            total_images += 1
            is_valid, reason = check_image_quality(img_path)
            
            if not is_valid:
                # Pindahkan ke folder rejected
                rejected_folder = rejected_path / class_folder
                rejected_folder.mkdir(exist_ok=True)
                
                dest = rejected_folder / img_path.name
                shutil.move(str(img_path), str(dest))
                
                print(f"❌ REJECTED: {img_path.name} - {reason}")
                class_rejected += 1
                total_rejected += 1
        
        print(f"✓ {class_folder}: {len(images) - class_rejected} OK, {class_rejected} rejected")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"Total images checked: {total_images}")
    print(f"Total rejected: {total_rejected} ({total_rejected/total_images*100:.2f}%)")
    print(f"Total accepted: {total_images - total_rejected}")
    print("=" * 70)

# ============================================================================
# JALANKAN SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Path ke folder raw dataset
    RAW_DATASET_PATH = "../1_dataset/raw"
    
    print("Starting quality control...")
    print(f"Dataset path: {RAW_DATASET_PATH}\n")
    
    quality_control_dataset(RAW_DATASET_PATH)
    
    print("\n✓ Quality control selesai!")
    print("Cek folder '1_dataset/rejected' untuk melihat foto yang ditolak.")