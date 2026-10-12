# Breast Cancer Detection using Pretrained Deep Learning Models

## Project Overview
This project implements an automated breast cancer detection system using Transfer Learning with pretrained deep learning models (ResNet50, VGG16) on histopathology images.

## Project Structure
```
CNN/
├── data/                    # Dataset directory (create this and add your data)
│   ├── 0/                  # Benign images
│   └── 1/                  # Malignant images
├── models/                  # Saved models and training metrics
├── src/                     # Source code
│   ├── config.py           # Configuration and hyperparameters
│   ├── preprocessing.py    # Data transformations
│   ├── data_loader.py      # Dataset and DataLoader classes
│   ├── model.py            # Model architecture
│   └── train.py            # Training script
├── requirements.txt         # Python dependencies
├── Problem_Statement.md     # Project problem statement
└── README.md               # This file
```

## Installation

### 1. Install Dependencies
From the project root directory (`CNN/`), run:
```bash
pip install -r requirements.txt
```

**Note:** The `requirements.txt` is in the root directory, not in `src/`.

### 2. Prepare Your Dataset
Download a breast cancer histopathology dataset (e.g., from Kaggle: [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)).

Organize your data in one of these structures:

**Option 1: Simple class folders**
```
data/
├── 0/          # Benign images
│   ├── img1.png
│   ├── img2.png
│   └── ...
└── 1/          # Malignant images
    ├── img1.png
    ├── img2.png
    └── ...
```

**Option 2: Kaggle IDC structure** (automatically detected)
```
data/
├── patient_id_1/
│   ├── 0/
│   │   └── images...
│   └── 1/
│       └── images...
└── patient_id_2/
    └── ...
```

## Usage

### Training the Model
From the project root directory:
```bash
python src/train.py
```

This will:
1. Load and preprocess the data from `data/` directory
2. Split into train/val/test sets (70%/10%/20%)
3. Train a ResNet50 model with pretrained ImageNet weights
4. Save the best model to `models/best_model.pth`
5. Generate training metrics plot at `models/training_metrics.png`
6. Evaluate on the test set

### Configuration
Edit `src/config.py` to customize:
- `BATCH_SIZE`: Batch size for training (default: 32)
- `NUM_EPOCHS`: Number of training epochs (default: 5)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `IMG_SIZE`: Image size (default: 224 for ResNet/VGG)
- `DATA_DIR`: Path to your dataset

### Using Different Models
In `src/train.py`, change the model initialization:
```python
# For ResNet50 (default)
model = BreastCancerClassifier(model_name='resnet50', num_classes=2, freeze_backbone=True)

# For VGG16
model = BreastCancerClassifier(model_name='vgg16', num_classes=2, freeze_backbone=True)
```

### Testing with Dummy Data
To verify the pipeline works:
```bash
python src/create_dummy_data.py
python src/train.py
```

## Key Features
- **Transfer Learning**: Uses pretrained ImageNet weights
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Automatic Data Splitting**: Stratified train/val/test splits
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Metrics Visualization**: Plots training/validation loss and accuracy

## Expected Output
After training completes, you'll see:
- Training progress for each epoch
- Best validation accuracy
- Test set accuracy
- Saved model at `models/best_model.pth`
- Training metrics plot at `models/training_metrics.png`

## Troubleshooting

### "No images found"
- Ensure your `data/` directory exists and contains images
- Check that images are organized in `0/` and `1/` subdirectories

### "ModuleNotFoundError"
- Make sure you installed dependencies: `pip install -r requirements.txt`
- Run from the project root directory, not from `src/`

### Low accuracy on dummy data
- This is expected! Dummy data is random noise
- Use real histopathology images for actual training

## Next Steps
1. Download a real dataset (e.g., Kaggle Breast Histopathology)
2. Extract it to the `data/` directory
3. Adjust `NUM_EPOCHS` in `config.py` (recommend 20-50 for real training)
4. Run `python src/train.py`
5. Monitor training metrics and adjust hyperparameters as needed

## References
- Problem Statement: See `Problem_Statement.md`
- Model Architecture: See `src/model.py`
- Data Pipeline: See `src/data_loader.py` and `src/preprocessing.py`
