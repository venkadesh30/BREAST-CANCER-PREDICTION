import torch
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
SEED = 42

# Learning rate scheduler
LR_STEP_SIZE = 7  # Decay LR every 7 epochs
LR_GAMMA = 0.1    # Multiply LR by 0.1 at each step

# Early stopping
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs

# Image config
IMG_SIZE = 224  # ResNet standard
CHANNELS = 3

# Compute device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
