import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import glob
import numpy as np

import config
import preprocessing

class HistopathologyDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # handle error gracefully? for now, just skip or return a dummy
            # but usually better to fail fast or fix data
            return torch.zeros((3, config.IMG_SIZE, config.IMG_SIZE)), torch.tensor(label, dtype=torch.long)

def get_data_loaders(data_dir=config.DATA_DIR):
    """
    Scans data_dir for images and creates train/val/test loaders.
    Assumes structure: 
    data_dir/
      0/ (benign)
      1/ (malignant)
    OR
    Recursive search for pattern (e.g. Kaggle dataset often has patient_id/0/img.png)
    """
    
    # Flexible search: find all png/jpg/jpeg files
    # Check if standard class folders exist
    class_0_path = os.path.join(data_dir, '0')
    class_1_path = os.path.join(data_dir, '1')
    
    all_files = []
    all_labels = []

    # Strategy 1: Standard ImageFolder structure
    if os.path.isdir(class_0_path) and os.path.isdir(class_1_path):
        print("Found class folders '0' and '1'.")
        # Class 0
        files_0 = glob.glob(os.path.join(class_0_path, '*.*')) # catch all extensions
        all_files.extend(files_0)
        all_labels.extend([0] * len(files_0))
        
        # Class 1
        files_1 = glob.glob(os.path.join(class_1_path, '*.*'))
        all_files.extend(files_1)
        all_labels.extend([1] * len(files_1))
        
    else:
        # Strategy 2: Recursive search (Kaggle IDC structure)
        # e.g. 8863/0/8863_idx5_x1001_y1001_class0.png
        print("Standard class folders not found. Searching recursively...")
        # Search for files ending in class0.png or class1.png which is common in IDC dataset
        # OR just find parent folder name
        
        # Taking a simpler approach: look for all images and decide label by parent folder or filename
        # Pattern: .../0/*.png -> 0, .../1/*.png -> 1
        
        import pathlib
        data_root = pathlib.Path(data_dir)
        # find all images
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif']
        found_images = []
        for ext in image_extensions:
            found_images.extend(list(data_root.rglob(ext)))
            
        print(f"Found {len(found_images)} images recursively.")
        
        for img_path in found_images:
            # Check parent folder for class
            parent = img_path.parent.name
            if parent == '0':
                all_files.append(str(img_path))
                all_labels.append(0)
            elif parent == '1':
                all_files.append(str(img_path))
                all_labels.append(1)
            # Check filename for class (common in some datasets)
            elif 'class0' in img_path.name:
                all_files.append(str(img_path))
                all_labels.append(0)
            elif 'class1' in img_path.name:
                all_files.append(str(img_path))
                all_labels.append(1)

    if not all_files:
        print("No images found! Please check data directory.")
        return None, None, None

    print(f"Total images: {len(all_files)}")
    print(f"Class distribution: {Counter(all_labels)}")

    # Split: Train+Val vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_files, all_labels, test_size=config.TEST_SPLIT, stratify=all_labels, random_state=config.SEED
    )

    # Split: Train vs Val
    # Adjust val split relative to train_val size
    val_split_adjusted = config.VAL_SPLIT / (1 - config.TEST_SPLIT) 
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split_adjusted, stratify=y_train_val, random_state=config.SEED
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create Datasets
    train_dataset = HistopathologyDataset(X_train, y_train, transform=preprocessing.get_transforms('train'))
    val_dataset = HistopathologyDataset(X_val, y_val, transform=preprocessing.get_transforms('val'))
    test_dataset = HistopathologyDataset(X_test, y_test, transform=preprocessing.get_transforms('test'))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test block
    loaders = get_data_loaders()
    if loaders[0] is not None:
        train_loader, _, _ = loaders
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
