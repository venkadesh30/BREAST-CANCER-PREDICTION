import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import config
import random


def create_realistic_benign(size=50):
    """Generate a synthetic benign tissue-like image.
    Benign tissue: smooth, uniform pink/purple staining with regular cell arrangement.
    """
    # Base pink/light purple tissue color
    base_r = random.randint(200, 240)
    base_g = random.randint(160, 200)
    base_b = random.randint(190, 230)

    img_array = np.zeros((size, size, 3), dtype=np.uint8)
    img_array[:, :, 0] = base_r
    img_array[:, :, 1] = base_g
    img_array[:, :, 2] = base_b

    # Add smooth texture noise
    noise = np.random.normal(0, 8, (size, size, 3)).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Draw regular, evenly-spaced small nuclei (benign pattern)
    num_cells = random.randint(8, 20)
    for _ in range(num_cells):
        cx = random.randint(3, size - 4)
        cy = random.randint(3, size - 4)
        radius = random.randint(1, 3)
        # Dark purple nuclei
        color = (
            random.randint(80, 130),
            random.randint(50, 100),
            random.randint(100, 150),
        )
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    # Smooth the image slightly
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def create_realistic_malignant(size=50):
    """Generate a synthetic malignant tissue-like image.
    Malignant tissue: dense, irregular dark clusters with high nuclear-to-cytoplasm ratio.
    """
    # Base darker, more variable tissue color
    base_r = random.randint(170, 210)
    base_g = random.randint(120, 170)
    base_b = random.randint(160, 200)

    img_array = np.zeros((size, size, 3), dtype=np.uint8)
    img_array[:, :, 0] = base_r
    img_array[:, :, 1] = base_g
    img_array[:, :, 2] = base_b

    # Add more aggressive texture noise (irregular staining)
    noise = np.random.normal(0, 18, (size, size, 3)).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Draw dense, irregular, overlapping nuclei (malignant pattern)
    num_cells = random.randint(25, 50)
    for _ in range(num_cells):
        cx = random.randint(2, size - 3)
        cy = random.randint(2, size - 3)
        radius = random.randint(2, 5)
        # Very dark purple/blue nuclei (high staining)
        color = (
            random.randint(40, 90),
            random.randint(20, 60),
            random.randint(70, 120),
        )
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    # Add some dark irregular patches (necrosis-like regions)
    num_patches = random.randint(1, 4)
    for _ in range(num_patches):
        px = random.randint(5, size - 10)
        py = random.randint(5, size - 10)
        pw = random.randint(5, 15)
        ph = random.randint(5, 15)
        patch_color = (
            random.randint(60, 100),
            random.randint(30, 70),
            random.randint(80, 120),
        )
        draw.ellipse([px, py, px + pw, py + ph], fill=patch_color)

    return img


def create_dummy_data(num_images=200):
    """Generate realistic synthetic histopathology images for training."""
    data_dir = config.DATA_DIR

    # Clear old images
    for cls in ['0', '1']:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.exists(cls_dir):
            for f in os.listdir(cls_dir):
                os.remove(os.path.join(cls_dir, f))
        os.makedirs(cls_dir, exist_ok=True)

    # Generate benign images (class 0)
    cls_dir_0 = os.path.join(data_dir, '0')
    print(f"Generating {num_images} benign images...")
    for i in range(num_images):
        img = create_realistic_benign(size=50)
        img.save(os.path.join(cls_dir_0, f"benign_{i:04d}.png"))

    # Generate malignant images (class 1)
    cls_dir_1 = os.path.join(data_dir, '1')
    print(f"Generating {num_images} malignant images...")
    for i in range(num_images):
        img = create_realistic_malignant(size=50)
        img.save(os.path.join(cls_dir_1, f"malignant_{i:04d}.png"))

    print(f"✅ Generated {num_images * 2} total images ({num_images} per class)")
    print(f"   Benign:    {cls_dir_0}")
    print(f"   Malignant: {cls_dir_1}")


if __name__ == "__main__":
    create_dummy_data()
