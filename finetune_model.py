"""
This script is meant to be run in Google Colab for fine-tuning the YOLOv10 model on the BCCD dataset.
It contains all the steps needed for training and should be run before deploying the application.
"""

import os
import glob
import zipfile
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import ultralytics
from ultralytics import YOLO
import numpy as np
import time

def download_bccd_dataset():
    """
    Downloads the BCCD dataset from the GitHub repository.
    Returns the path to the dataset directory.
    """
    # Install dependencies if needed
    os.system('pip install ultralytics gdown')
    
    # Clone the repository
    os.system('git clone https://github.com/Shenggan/BCCD_Dataset.git')
    
    # Verify download
    dataset_dir = Path('BCCD_Dataset')
    if not dataset_dir.exists():
        print("Failed to download dataset using git. Trying alternative download...")
        # Alternative download method using direct download links
        os.makedirs('BCCD_Dataset/BCCD', exist_ok=True)
        url = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"
        r = requests.get(url, allow_redirects=True)
        with open('bccd_dataset.zip', 'wb') as f:
            f.write(r.content)
        
        # Extract the zipfile
        with zipfile.ZipFile('bccd_dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Move contents to the expected location
        extracted_dir = Path('BCCD_Dataset-master')
        if extracted_dir.exists():
            # Copy contents to the BCCD_Dataset directory
            for item in extracted_dir.glob('*'):
                if item.is_dir():
                    shutil.copytree(item, dataset_dir / item.name)
                else:
                    shutil.copy(item, dataset_dir / item.name)
    
    print("Dataset downloaded successfully.")
    return dataset_dir

def setup_dataset_for_yolo(dataset_path):
    """
    Prepares the BCCD dataset for YOLO format.
    Args:
        dataset_path: Path to the downloaded dataset
    Returns:
        Path to the processed dataset
    """
    yolo_dir = Path('BCCD_YOLO')
    os.makedirs(yolo_dir, exist_ok=True)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(yolo_dir / split / 'images', exist_ok=True)
        os.makedirs(yolo_dir / split / 'labels', exist_ok=True)
    
    # Map sources to destinations
    splits = {
        'train': dataset_path / 'BCCD' / 'train',
        'val': dataset_path / 'BCCD' / 'val',
        'test': dataset_path / 'BCCD' / 'test'
    }
    
    # Process each split
    for split_name, split_dir in splits.items():
        image_files = list(split_dir.glob('*.jpg'))
        for img_file in image_files:
            # Copy image
            shutil.copy(img_file, yolo_dir / split_name / 'images' / img_file.name)
            
            # Convert annotation
            xml_file = split_dir / f"{img_file.stem}.xml"
            if xml_file.exists():
                txt_file = yolo_dir / split_name / 'labels' / f"{img_file.stem}.txt"
                convert_annotations(xml_file, txt_file)
    
    return yolo_dir

def convert_annotations(xml_path, txt_path):
    """
    Converts XML annotations to YOLO format TXT files.
    Args:
        xml_path: Path to XML annotation file
        txt_path: Path to output TXT file
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Map class names to IDs
    class_map = {'RBC': 0, 'WBC': 1, 'Platelets': 2}
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_map:
                continue
                
            cls_id = class_map[cls_name]
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
            
            # Convert to YOLO format: center_x, center_y, width, height
            x_center = (x_min + x_max) / (2.0 * img_width)
            y_center = (y_min + y_max) / (2.0 * img_height)
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Write to file
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

def create_dataset_yaml(dataset_path):
    """
    Creates the YAML file required by YOLOv10 for training.
    Args:
        dataset_path: Path to the processed dataset
    """
    yaml_content = f"""
# YOLOv10 dataset config for BCCD
path: {dataset_path.absolute()}  # Root directory
train: train/images  # Train images relative to path
val: val/images      # Validation images relative to path
test: test/images    # Test images relative to path

# Classes
names:
  0: RBC
  1: WBC
  2: Platelets

# Number of classes
nc: 3
"""
    
    yaml_path = dataset_path / 'bccd.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def train_model(dataset_path):
    """
    Trains YOLOv10 on the BCCD dataset.
    Args:
        dataset_path: Path to the processed dataset
    Returns:
        Path to the trained model
    """
    # Create YAML config file
    yaml_path = create_dataset_yaml(dataset_path)
    
    # Import required modules
    import torch
    
    # Load a pretrained YOLOv10 model
    # Note: Use 'yolov10n.pt' for faster training, or 'yolov10s.pt' for better accuracy
    model = YOLO('yolov10n.pt')  # Nano model
    
    # Train the model
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    results = model.train(
        data=str(yaml_path),
        epochs=50,           # Number of epochs
        imgsz=640,           # Image size
        batch=16,            # Batch size
        patience=15,         # Early stopping patience
        device=device,
        project='BCCD_Training',
        name='yolov10_bccd',
        seed=42,
        workers=8 if torch.cuda.is_available() else 1
    )
    
    # Get the path to the best model
    best_model_path = Path('BCCD_Training/yolov10_bccd/weights/best.pt')
    
    # Export the model to other formats if needed
    model.export(format='onnx')
    
    # Copy model to Google Drive if running in Colab
    try:
        from google.colab import drive
        drive_path = Path('/content/drive/MyDrive/BCCD_Model')
        drive_path.mkdir(exist_ok=True, parents=True)
        
        model_save_path = drive_path / 'yolov10_bccd.pt'
        shutil.copy(best_model_path, model_save_path)
        print(f"Model saved to Google Drive at {model_save_path}")
    except:
        print("Not running in Colab or couldn't mount Google Drive.")
    
    return best_model_path

def main():
    """
    Main function to execute the fine-tuning process.
    """
    start_time = time.time()
    
    print("Step 1: Downloading BCCD dataset...")
    dataset_path = download_bccd_dataset()
    
    print("Step 2: Setting up dataset in YOLO format...")
    yolo_dataset_path = setup_dataset_for_yolo(dataset_path)
    
    print("Step 3: Training YOLOv10 model...")
    trained_model_path = train_model(yolo_dataset_path)
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed_time:.2f} minutes.")
    print(f"Trained model saved at: {trained_model_path}")
    
    return trained_model_path

if __name__ == "__main__":
    main()