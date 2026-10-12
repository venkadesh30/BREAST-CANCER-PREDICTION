# Troubleshooting Guide

## IDE Warnings vs. Actual Errors

### ✅ Your Code is Working Fine!

All Python files have been tested and work correctly:
- ✅ `src/config.py` - No errors
- ✅ `src/preprocessing.py` - No errors  
- ✅ `src/data_loader.py` - No errors
- ✅ `src/model.py` - No errors
- ✅ `src/train.py` - No errors

### 🔍 What You're Seeing

The "errors" in your IDE are **linter warnings**, not actual Python errors. They appear because:

1. **Import warnings** - The IDE can't find `torch`, `numpy`, etc. in its search path
2. **Module path warnings** - Files in `src/` import each other (e.g., `import config`)

**These warnings do NOT affect code execution!**

## How to Fix IDE Warnings

### Option 1: Select Python Interpreter (VS Code)

1. Press `Ctrl + Shift + P`
2. Type "Python: Select Interpreter"
3. Choose the Python installation where you ran `pip install -r requirements.txt`
   - Usually: `C:\Users\selva\AppData\Local\Programs\Python\Python311\python.exe`

### Option 2: Reload Window (VS Code)

1. Press `Ctrl + Shift + P`
2. Type "Developer: Reload Window"
3. Wait for the window to reload

### Option 3: Install Python Extension (VS Code)

1. Go to Extensions (`Ctrl + Shift + X`)
2. Search for "Python" by Microsoft
3. Install it
4. Restart VS Code

## Verification Tests

Run these commands to verify everything works:

```bash
# Test all imports
python -c "import sys; sys.path.insert(0, 'src'); import config; import preprocessing; import data_loader; import model; print('✅ All imports successful!')"

# Test model architecture
python src/model.py

# Test data loading
python src/data_loader.py

# Test full training (5 epochs on dummy data)
python src/train.py
```

## Common Issues

### "ModuleNotFoundError: No module named 'torch'"
**Solution:** Install dependencies from project root:
```bash
cd C:\Users\selva\OneDrive\Desktop\CNN
pip install -r requirements.txt
```

### "No images found"
**Solution:** The dummy data is already created in `data/0/` and `data/1/`. If you deleted it:
```bash
python src/create_dummy_data.py
```

### Red squiggly lines in IDE
**Solution:** These are just warnings, not errors. The code runs fine. To remove them:
- Select the correct Python interpreter (see Option 1 above)
- Or ignore them - they don't affect execution

## Quick Reference

### Project Structure
```
CNN/
├── src/
│   ├── __init__.py          # Makes src a Python package
│   ├── config.py            # Configuration
│   ├── preprocessing.py     # Data transforms
│   ├── data_loader.py       # Dataset handling
│   ├── model.py             # Model architecture
│   └── train.py             # Training script
├── data/                     # Dataset
├── models/                   # Saved models
├── requirements.txt          # Dependencies
└── inference.py             # Prediction script
```

### Running Scripts

**Always run from project root (`CNN/` directory):**

```bash
# Training
python src/train.py

# Inference
python inference.py data/0/sample_image.png

# Create dummy data
python src/create_dummy_data.py
```

## Still Having Issues?

If you encounter actual runtime errors (not IDE warnings):

1. **Check you're in the correct directory:**
   ```bash
   cd C:\Users\selva\OneDrive\Desktop\CNN
   ```

2. **Verify Python version:**
   ```bash
   python --version
   # Should be Python 3.8 or higher
   ```

3. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

4. **Check the specific error message** and share it for debugging

---

**Bottom Line:** Your code is working perfectly! The IDE warnings are cosmetic and can be safely ignored. ✅
