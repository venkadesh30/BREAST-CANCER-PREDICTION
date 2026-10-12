# 🚀 Quick Start - Frontend Setup

## Choose Your Frontend

### 🎯 Fastest Option: Streamlit (Recommended for First-Time Users)

```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Browser opens automatically at http://localhost:8501
```

**That's it!** You can now:
- ✅ Upload histopathology images
- ✅ Get instant predictions (Benign/Malignant)
- ✅ View confidence scores
- ✅ Analyze multiple images at once
- ✅ Download analysis reports

---

### 🌐 Professional Option: Flask Web App

```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Run the Flask app
python web_app.py

# 3. Open http://localhost:5000 in your browser
```

**Features:**
- ✅ Modern, responsive UI
- ✅ REST API for integration
- ✅ Batch processing
- ✅ Statistics & history
- ✅ Production-ready

---

## First Time Setup Checklist

- [ ] Have you trained the model? (`python src/train.py`)
- [ ] Do you have `models/best_model.pth`?
- [ ] Installed requirements? (`pip install -r requirements.txt`)
- [ ] Have test images in `data/` folder?

If any of the above are "No", complete them first!

---

## 5-Minute Test Run

### Using Streamlit:

```bash
# Terminal 1: Start Streamlit
streamlit run app.py

# Browser automatically opens...
# 1. Go to "Diagnosis" tab
# 2. Click "Upload an image"
# 3. Select an image from data/0/ or data/1/
# 4. Click "Analyze Image"
# 5. See results!
```

### Using Flask:

```bash
# Terminal 1: Start Flask
python web_app.py

# Browser: Go to http://localhost:5000
# 1. Click upload area or drag-drop image
# 2. Click "Analyze Image"
# 3. See results immediately!
```

---

## Understanding Your First Result

### Example Output:

```
✅ BENIGN
Confidence: 89.34%
Risk Level: 🟢 LOW RISK

Benign Probability: 89.34%
Malignant Probability: 10.66%
```

**What This Means:**
- Model predicts: **No cancer** (Benign)
- Confidence: **89% sure**
- Risk: **Very low** - Can safely say it's benign

---

### Another Example:

```
⚠️ MALIGNANT
Confidence: 76.25%
Risk Level: 🟠 MEDIUM RISK

Benign Probability: 23.75%
Malignant Probability: 76.25%
```

**What This Means:**
- Model predicts: **Cancer detected** (Malignant)
- Confidence: **76% sure**
- Risk: **Medium** - Should be reviewed by expert

---

## Common Tasks

### Upload Single Image
**Streamlit**: 📤 Single Image tab → Upload → Analyze  
**Flask**: Click upload area → Analyze Image

### Batch Process Multiple Images
**Streamlit**: 📁 Batch Analysis tab → Select multiple → Analyze  
**Flask**: 📁 Batch Analysis tab → Select multiple → Analyze Batch

### View History
**Streamlit**: 📚 About → Recent Predictions  
**Flask**: 📊 Statistics tab → Scroll down for history

### Download Report
**Streamlit**: After analysis → Click "Download Report (TXT)"  
**Flask**: After analysis → Report downloads automatically

### See Model Info
**Streamlit**: 📊 Model Info tab → See full specifications  
**Flask**: About section → Model Information table

---

## Keyboard Shortcuts (Streamlit)

| Shortcut | Action |
|----------|--------|
| `C` | Clear cache |
| `R` | Rerun |
| `K` | Open keyboard help |

---

## Stop the Application

```bash
# In terminal window running the app:
# Press: Ctrl + C

# App will stop in ~2 seconds
```

---

## Change Port (If Default is Busy)

### Streamlit:
```bash
streamlit run app.py --server.port=8502
# Open: http://localhost:8502
```

### Flask:
```bash
# Edit web_app.py, last line:
# Change: app.run(debug=True, port=5000)
# To:     app.run(debug=True, port=5001)
```

---

## Use GPU for Faster Processing

If you have an NVIDIA GPU:

```bash
# The system automatically uses GPU if available
# If CUDA-enabled GPU detected, it will be used

# Check if GPU is being used in the app
# Look for: Device: cuda:0 in Model Info
```

If using CPU (slower but works fine):

```bash
# Already configured to use CPU automatically
# On average GPU is ~5-10x faster
```

---

## Test with Sample Images

### Using Provided Data:

```bash
# Sample images are in:
data/0/  # Benign examples
data/1/  # Malignant examples

# Test with a few of these in either app
```

---

## Troubleshooting Quick Fixes

### "Model not found" error
```bash
# Train the model first:
python src/train.py

# This creates: models/best_model.pth
```

### Streamlit won't open
```bash
# Try manually opening:
# http://localhost:8501
```

### Flask won't start
```bash
# Make sure port 5000 is free:
# netstat -ano | findstr :5000  (Windows)
# lsof -i :5000  (Mac/Linux)

# Or use different port (see above)
```

### Application is slow
```bash
# Check if GPU is enabled (Model Info)
# Consider using Streamlit (lighter) instead of Flask
# Close other applications
```

---

## Next Steps After First Run

1. **Explore the statistics** - See patterns in predictions
2. **Try batch processing** - Analyze multiple images
3. **Check model details** - Understand how transfer learning works
4. **Download reports** - Save analysis results
5. **Read documentation** - See FRONTEND_GUIDE.md for details

---

## Important Reminders ⚠️

- **NOT for clinical diagnosis** - This is educational only
- **Always consult experts** - Pathologists must validate results
- **Check confidence** - Low confidence (<60%) needs expert review
- **Keep privacy** - Never upload real patient data without consent
- **Understand errors** - AI can make mistakes

---

## Get Help

1. **Streamlit docs**: https://docs.streamlit.io/
2. **Flask docs**: https://flask.palletsprojects.com/
3. **PyTorch docs**: https://pytorch.org/docs/
4. **Check TROUBLESHOOTING.md** in this project

---

## That's It! 🎉

You now have a working breast cancer detection frontend!

**Choose one:**
- Fast & simple? → Streamlit: `streamlit run app.py`
- Professional & scalable? → Flask: `python web_app.py`

Both work great. Pick your favorite and start analyzing! 🏥

---

**Questions? Problems?**
Check FRONTEND_GUIDE.md for detailed explanations and troubleshooting.
