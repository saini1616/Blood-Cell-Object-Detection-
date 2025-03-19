"""
Module for preprocessing and data augmentation of the BCCD dataset.
"""
import os
import shutil
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import albumentations as A
from PIL import Image

def preprocess_dataset(dataset_path):
    """
    Preprocesses the BCCD dataset images.
    
    Args:
        dataset_path: Path to the BCCD dataset
    
    Returns:
        Path to the preprocessed dataset
    """
    print("Preprocessing dataset...")
    
    # Create output directory
    output_dir = "preprocessed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    # Get paths
    image_path = os.path.join(dataset_path, "BCCD", "JPEGImages")
    annot_path = os.path.join(dataset_path, "BCCD", "Annotations")
    
    # Get all image files
    image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for file in tqdm(image_files, desc="Preprocessing images"):
        # Read image
        img = cv2.imread(os.path.join(image_path, file))
        
        if img is None:
            continue
        
        # Apply preprocessing
        # 1. Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Normalize
        img = img / 255.0
        
        # 3. Resize to a standard size if needed
        img = cv2.resize(img, (640, 640))
        
        # 4. Convert back to 0-255 range and BGR for saving
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save the preprocessed image
        cv2.imwrite(os.path.join(output_dir, "images", file), img)
        
        # Copy the annotation file
        base_name = os.path.splitext(file)[0]
        xml_file = os.path.join(annot_path, base_name + ".xml")
        if os.path.exists(xml_file):
            shutil.copy(xml_file, os.path.join(output_dir, "annotations", base_name + ".xml"))
    
    print(f"Preprocessing completed. Saved to {output_dir}")
    return output_dir

def augment_dataset(dataset_path):
    """
    Applies data augmentation to the preprocessed dataset.
    
    Args:
        dataset_path: Path to the preprocessed dataset
    
    Returns:
        Path to the augmented dataset
    """
    print("Augmenting dataset...")
    
    # Create output directory
    output_dir = "augmented_dataset"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    # Copy original data first
    image_path = os.path.join(dataset_path, "images")
    annot_path = os.path.join(dataset_path, "annotations")
    
    for file in os.listdir(image_path):
        shutil.copy(os.path.join(image_path, file), 
                    os.path.join(output_dir, "images", file))
    
    for file in os.listdir(annot_path):
        shutil.copy(os.path.join(annot_path, file), 
                    os.path.join(output_dir, "annotations", file))
    
    # Define augmentation pipeline
    augmentations = [
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.BBoxParams(format='pascal_voc', label_fields=['class_labels'])
        ]),
        A.Compose([
            A.RandomBrightnessContrast(p=1.0),
            A.BBoxParams(format='pascal_voc', label_fields=['class_labels'])
        ]),
        A.Compose([
            A.Rotate(limit=20, p=1.0),
            A.BBoxParams(format='pascal_voc', label_fields=['class_labels'])
        ]),
        A.Compose([
            A.RandomSizedBBoxSafeCrop(width=640, height=640, p=1.0),
            A.BBoxParams(format='pascal_voc', label_fields=['class_labels'])
        ])
    ]
    
    # Get all image files
    image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for file in tqdm(image_files, desc="Augmenting images"):
        # Read image
        img_path = os.path.join(image_path, file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read annotation
        base_name = os.path.splitext(file)[0]
        xml_path = os.path.join(annot_path, base_name + ".xml")
        
        if not os.path.exists(xml_path):
            continue
        
        # Parse XML to get bounding boxes
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        bboxes = []
        class_labels = []
        
        for obj in root.findall('object'):
            cls = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            bboxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(cls)
        
        # Apply each augmentation
        for i, aug in enumerate(augmentations):
            # Apply augmentation
            try:
                augmented = aug(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
                
                # Skip if no bounding boxes are left after augmentation
                if len(aug_bboxes) == 0:
                    continue
                
                # Create a new XML file for the augmented image
                new_file_name = f"{base_name}_aug_{i}.jpg"
                new_xml_name = f"{base_name}_aug_{i}.xml"
                
                # Save the augmented image
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, "images", new_file_name), aug_img_bgr)
                
                # Create a new XML annotation
                create_xml_annotation(
                    os.path.join(output_dir, "annotations", new_xml_name),
                    new_file_name,
                    aug_img.shape[1],  # width
                    aug_img.shape[0],  # height
                    aug_bboxes,
                    aug_labels
                )
            except Exception as e:
                print(f"Error augmenting {file} with {i}: {e}")
                continue
    
    print(f"Augmentation completed. Saved to {output_dir}")
    return output_dir

def create_xml_annotation(path, filename, width, height, bboxes, labels):
    """
    Creates an XML annotation file in Pascal VOC format.
    
    Args:
        path: Path to save the XML file
        filename: Image filename
        width: Image width
        height: Image height
        bboxes: List of bounding boxes [xmin, ymin, xmax, ymax]
        labels: List of class labels
    """
    root = ET.Element("annotation")
    
    # Add image information
    folder = ET.SubElement(root, "folder")
    folder.text = "images"
    
    file_node = ET.SubElement(root, "filename")
    file_node.text = filename
    
    size = ET.SubElement(root, "size")
    width_node = ET.SubElement(size, "width")
    width_node.text = str(width)
    height_node = ET.SubElement(size, "height")
    height_node.text = str(height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    
    # Add object information
    for (xmin, ymin, xmax, ymax), label in zip(bboxes, labels):
        obj = ET.SubElement(root, "object")
        
        name = ET.SubElement(obj, "name")
        name.text = label
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin_node = ET.SubElement(bndbox, "xmin")
        xmin_node.text = str(int(xmin))
        ymin_node = ET.SubElement(bndbox, "ymin")
        ymin_node.text = str(int(ymin))
        xmax_node = ET.SubElement(bndbox, "xmax")
        xmax_node.text = str(int(xmax))
        ymax_node = ET.SubElement(bndbox, "ymax")
        ymax_node.text = str(int(ymax))
    
    # Write to file
    tree = ET.ElementTree(root)
    tree.write(path)
