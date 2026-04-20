# 🎭 Meme Bully Classifier - Friend Setup Instructions

**For: Dhruv Garg & Dibya**

Hi! This guide tells you what YOU need to do to complete the project.

---

## ⚡ Quick Summary

**You need to:**
1. ✅ **Add meme images** to the `data/` folder
2. ✅ **Train the CNN model** using `train_cnn.py`
3. ✅ **Test the app** with your trained model

**Then it's ready!**

---

## Step 1️⃣: Add Your Meme Images

### Create the data folder structure:

```
MemeClassifier_Project/
├── data/
│   ├── bully/
│   │   ├── meme1.jpg
│   │   ├── meme2.jpg
│   │   ├── meme3.jpg
│   │   └── ... (add all bully memes here)
│   └── non_bully/
│       ├── meme1.jpg
│       ├── meme2.jpg
│       ├── meme3.jpg
│       └── ... (add all non-bully memes here)
```

### Where to put memes:
- **Bully memes** → `data/bully/` folder
- **Non-bully memes** → `data/non_bully/` folder

### How many images?
- **Minimum:** 50 images per class (100 total)
- **Better:** 100-200 images per class (200-400 total)
- **Best:** 500+ images per class

### Image requirements:
- Formats: `.jpg`, `.png`, `.gif`, `.bmp`
- Any size (model resizes to 224x224)
- Clear, readable memes
- Balanced (roughly equal bullies vs non-bullies)

---

## Step 2️⃣: Train the CNN Model

### Open PowerShell and navigate to project:

```powershell
cd C:\Users\aryan\Desktop\MemeClassifier_Project
```

### Create virtual environment:

```powershell
python -m venv train_env
train_env\Scripts\Activate.ps1
```

### Install dependencies:

```powershell
pip install torch torchvision
pip install streamlit Pillow numpy
```

### Run training:

```powershell
python train_cnn.py
```

### What you'll see:

```
🎭 Meme Bully CNN Training
============================================================

📂 Loading images from data/...
✅ Loaded 150 images
   - Training: 120 images
   - Validation: 30 images

🖥️  Using device: cpu

🤖 Creating CNN Model (ResNet50 + Custom Head)...
✅ Model created

🏋️  Starting training...

Epoch 1/15
--------------------------------------------------
Train Loss: 0.6543, Train Acc: 65.23%
Val Loss: 0.5234, Val Acc: 72.50%
✅ Best model saved!

... (continues for all epochs)

✅ Training Complete!
Model saved: models/best_model.pth
```

**Training time:**
- **CPU:** 10-30 minutes
- **GPU (if available):** 2-5 minutes

---

## Step 3️⃣: Test the App

After training completes, run the app:

```powershell
streamlit run app.py
```

The app will open in your browser. Now you can:
1. ✅ Upload a meme image
2. ✅ Get classification (BULLY / NON-BULLY)
3. ✅ See confidence score
4. ✅ Rate the meme
5. ✅ Save results

---

## ✅ You're Done!

Once training completes and the app works:

1. Test with different meme images
2. Check accuracy and confidence scores
3. Save some results as examples
4. You're ready to submit! 🚀

---

## 📋 Checklist

- [ ] Created `data/bully/` folder with memes
- [ ] Created `data/non_bully/` folder with memes
- [ ] Have at least 100 images total (50+ per class)
- [ ] Installed PyTorch: `pip install torch torchvision`
- [ ] Ran training: `python train_cnn.py`
- [ ] Training completed successfully (see models/best_model.pth)
- [ ] Ran app: `streamlit run app.py`
- [ ] Tested classification with sample memes
- [ ] App shows real CNN predictions ✅

---

## 🆘 Troubleshooting

**"No images found"**
- Make sure folders exist: `data/bully/` and `data/non_bully/`
- Check images are in the right format (.jpg, .png, etc.)

**"Out of memory"**
- You have too many images or not enough RAM
- Reduce images or use smaller batches

**"Very low accuracy (< 50%)"**
- Not enough images (need 100+)
- Images are very different from typical memes
- Add more diverse training data

**"Training is very slow"**
- Normal on CPU (takes 10-30 min)
- Use GPU if available (much faster)

---

## 📚 More Info

For detailed information about:
- **CNN model details** → See `cnn_model.py`
- **Full training guide** → Read `TRAINING_GUIDE.md`
- **How to use the app** → See `HOW_TO_RUN.md`

---

## 🎯 Summary for You

**Your task in 3 steps:**

1. **Add images** to `data/bully/` and `data/non_bully/`
2. **Run** `python train_cnn.py`
3. **Test** `streamlit run app.py`

**That's it!** The rest is automatic. 

Good luck! 🚀

---

**Questions?** Ask Shreyan or check the main README.md

