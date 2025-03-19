"""
Script for fine-tuning YOLOv10 on the BCCD dataset.
This is designed to be run in Google Colab.
"""

import os
import shutil
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import random
import numpy as np

def download_bccd_dataset():
    """
    Download the BCCD dataset from the GitHub repository.
    """
    # Install necessary libraries
    os.system('pip install ultralytics')
    
    # Clone the repository containing the BCCD dataset
    os.system('git clone https://github.com/Shenggan/BCCD_Dataset.git')
    
    # Ensure directory exists
    data_dir = Path('BCCD_Dataset')
    if data_dir.exists():
        print(f"Dataset downloaded to {data_dir.absolute()}")
        return data_dir
    else:
        raise FileNotFoundError("Failed to download the dataset")

def prepare_yolo_format(data_dir):
    """
    Prepare the dataset in YOLO format.
    
    Args:
        data_dir (Path): Path to the BCCD dataset directory
    """
    # Create directories for YOLO format
    yolo_dir = Path('BCCD_YOLO')
    yolo_dir.mkdir(exist_ok=True)
    
    # Create train, val, test directories
    train_dir = yolo_dir / 'train'
    val_dir = yolo_dir / 'val'
    test_dir = yolo_dir / 'test'
    
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(exist_ok=True)
        (d / 'images').mkdir(exist_ok=True)
        (d / 'labels').mkdir(exist_ok=True)
    
    # Source directories
    train_src = data_dir / 'BCCD' / 'train'
    test_src = data_dir / 'BCCD' / 'test'
    val_src = data_dir / 'BCCD' / 'val'
    
    # Process training data
    process_dataset_split(train_src, train_dir)
    
    # Process validation data
    process_dataset_split(val_src, val_dir)
    
    # Process test data
    process_dataset_split(test_src, test_dir)
    
    return yolo_dir

def process_dataset_split(src_dir, dest_dir):
    """
    Process a dataset split (train, val, test) to YOLO format.
    
    Args:
        src_dir (Path): Source directory with images and annotations
        dest_dir (Path): Destination directory for YOLO format
    """
    # Copy images
    img_files = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
    for img_file in img_files:
        shutil.copy(img_file, dest_dir / 'images' / img_file.name)
    
    # Convert annotations
    xml_files = list(src_dir.glob('*.xml'))
    for xml_file in xml_files:
        txt_file = dest_dir / 'labels' / (xml_file.stem + '.txt')
        convert_annotation(xml_file, txt_file)

def convert_annotation(xml_path, output_path):
    """
    Convert XML annotation to YOLO format.
    
    Args:
        xml_path (Path): Path to XML annotation file
        output_path (Path): Path to output YOLO format file
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            # Convert class name to class id
            if class_name == 'RBC':
                class_id = 0
            elif class_name == 'WBC':
                class_id = 1
            elif class_name == 'Platelets':
                class_id = 2
            else:
                continue  # Skip unknown classes
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (centerx, centery, width, height)
            # All values normalized to [0, 1]
            x_center = (x_min + x_max) / (2.0 * w)
            y_center = (y_min + y_max) / (2.0 * h)
            bbox_width = (x_max - x_min) / w
            bbox_height = (y_max - y_min) / h
            
            # Write to file
            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def create_dataset_yaml(data_dir):
    """
    Create dataset YAML file for YOLO training.
    
    Args:
        data_dir (Path): Path to the dataset directory
    """
    yaml_content = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'RBC',
            1: 'WBC',
            2: 'Platelets'
        },
        'nc': 3  # Number of classes
    }
    
    with open(data_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return data_dir / 'dataset.yaml'

def apply_data_augmentation(data_dir):
    """
    Apply data augmentation to the training set.
    
    Args:
        data_dir (Path): Path to the dataset directory
    """
    # Here we would implement data augmentation techniques
    # This is a placeholder for real augmentation code
    print("Applying data augmentation...")
    
    # In a real implementation, you would:
    # 1. Load images
    # 2. Apply transformations (rotation, flip, color changes, etc.)
    # 3. Save augmented images with corresponding labels
    
    # For demonstration purposes only - not actual augmentation
    train_img_dir = data_dir / 'train' / 'images'
    train_lbl_dir = data_dir / 'train' / 'labels'
    
    print(f"Training images: {len(list(train_img_dir.glob('*.jpg')))}")
    print(f"Training labels: {len(list(train_lbl_dir.glob('*.txt')))}")
    
    print("Data augmentation complete.")

def train_yolov10(data_dir):
    """
    Fine-tune YOLOv10 on the BCCD dataset.
    
    Args:
        data_dir (Path): Path to the dataset directory
    """
    # Import ultralytics and train the model
    from ultralytics import YOLO
    
    # Load a pretrained YOLOv10 model
    model = YOLO('yolov10n.pt')  # Smaller model for faster training
    
    # Train the model with the BCCD dataset
    yaml_path = data_dir / 'dataset.yaml'
    results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        patience=10,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        name='yolov10_bccd'
    )
    
    # Save the model to Google Drive for safekeeping
    output_dir = Path('/content/drive/MyDrive/yolov10_bccd')
    output_dir.mkdir(exist_ok=True, parents=True)
    model.export(format='onnx')  # Export to ONNX format
    
    # Copy last trained weights
    best_pt = Path(f"runs/train/yolov10_bccd/weights/best.pt")
    if best_pt.exists():
        shutil.copy(best_pt, output_dir / 'yolov10_bccd_best.pt')
        print(f"Model saved to {output_dir / 'yolov10_bccd_best.pt'}")
    
    return results

def main():
    """
    Main function to execute the fine-tuning process.
    """
    # Mount Google Drive for saving the final model
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
    except:
        print("Warning: Not running in Google Colab, or Drive mounting failed.")
    
    # Download and prepare dataset
    data_dir = download_bccd_dataset()
    print("Dataset downloaded")
    
    # Prepare data in YOLO format
    yolo_dir = prepare_yolo_format(data_dir)
    print("Dataset prepared in YOLO format")
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml(yolo_dir)
    print(f"Dataset YAML created at {yaml_path}")
    
    # Apply data augmentation
    apply_data_augmentation(yolo_dir)
    
    # Train the model
    results = train_yolov10(yolo_dir)
    print("Training complete!")
    
    return results

if __name__ == "__main__":
    main()