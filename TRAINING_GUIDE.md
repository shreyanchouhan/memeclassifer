# 🤖 CNN Training Guide

**Train your Meme Bully Classifier with Deep Learning**

---

## Step 1️⃣: Prepare Your Data

Create this folder structure in your project:

```
MemeClassifier_Project/
├── data/
│   ├── bully/
│   │   ├── meme1.jpg
│   │   ├── meme2.jpg
│   │   └── ...
│   └── non_bully/
│       ├── meme1.jpg
│       ├── meme2.jpg
│       └── ...
├── train_cnn.py
├── cnn_model.py
└── app.py
```

**Tips for data:**
- At least **50 images** per class (100+ total)
- Mix of different meme styles/variations
- Balanced: roughly equal bully vs non-bully
- Various sizes (model resizes to 224x224)

---

## Step 2️⃣: Install Training Dependencies

```powershell
cd C:\Users\aryan\Desktop\MemeClassifier_Project

# Create fresh virtual environment
python -m venv train_env
train_env\Scripts\Activate.ps1

# Install PyTorch + Streamlit
pip install torch torchvision
pip install streamlit Pillow numpy
```

**Note:** PyTorch is large (~2GB). Takes few minutes to install.

---

## Step 3️⃣: Train the Model

```powershell
# Make sure you're in the venv
train_env\Scripts\Activate.ps1

# Run training
python train_cnn.py
```

**What happens:**
```
Loading images from data/...
✅ Loaded 120 images
   - Training: 96 images
   - Validation: 24 images

🖥️  Using device: cpu
(or cuda if you have GPU)

🤖 Creating CNN Model...
✅ Model created

🏋️  Starting training...

Epoch 1/15
--------------------------------------------------
Train Loss: 0.6543, Train Acc: 65.23%
Val Loss: 0.5234, Val Acc: 72.50%
✅ Best model saved!

... (continues for 15 epochs)

✅ Training Complete!
Model saved: models/best_model.pth
```

**Training takes:**
- **CPU:** 10-30 minutes (slow but works)
- **GPU:** 2-5 minutes (much faster)

---

## Step 4️⃣: Run the App with Trained Model

```powershell
# Still in venv
streamlit run app.py
```

**Now your app will:**
- ✅ Load the trained CNN model
- ✅ Use REAL deep learning for classification
- ✅ Show confidence scores
- ✅ Accept ratings and feedback

---

## Understanding the CNN

### **What the Model Does:**

```
Your Meme Image (224x224 pixels)
         ↓
   ResNet50 Base Model
   (Pre-trained on ImageNet)
   - Detects edges, shapes, objects
   - Recognizes common patterns
         ↓
   Custom Classification Head
   - Bully patterns detection
   - Non-bully patterns detection
         ↓
   Output: BULLY or NON-BULLY
   (with confidence %)
```

### **Why ResNet50?**
- Pre-trained on millions of images
- Fast transfer learning
- Good accuracy even with small datasets
- Perfect for course project

---

## Model Architecture

```python
MemeBullyCNN:
├── ResNet50 (frozen early layers)
├── Custom Head:
│   ├── Dense(512, ReLU)
│   ├── Dropout(0.5)
│   ├── Dense(256, ReLU)
│   ├── Dropout(0.3)
│   └── Dense(2, Softmax)  # Output: [Non-Bully, Bully]
└── Parameters: ~25M (mostly from ResNet50)
```

---

## Training Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | 15 | Complete passes through dataset |
| **Batch Size** | 32 | Images per training step |
| **Learning Rate** | 0.001 | How fast model learns |
| **Optimizer** | Adam | Modern gradient descent |
| **Loss Function** | CrossEntropyLoss | For classification |
| **Scheduler** | StepLR | Reduces LR every 5 epochs |

---

## Output Files

After training, you'll have:

```
models/
├── best_model.pth        ← Best model (highest validation accuracy)
├── final_model.pth       ← Model after all epochs
└── training_info.txt     ← Training statistics

results/
└── meme_20260420_*.json  ← Classification results from app
```

---

## Troubleshooting

### "Out of Memory" Error
```
Solution: Reduce batch_size in train_cnn.py
batch_size = 16  # or even 8
```

### "No Images Found"
```
✅ Check folder structure:
data/bully/          (must have images)
data/non_bully/      (must have images)
```

### "Module not found: torch"
```
Solution: Install PyTorch
pip install torch torchvision
```

### Very Low Accuracy (< 50%)
```
Reasons:
- Too few images (need 100+)
- Imbalanced data (more bullies than non-bullies)
- Images too different from training data

Solution: Add more diverse images
```

---

## Performance Expectations

With different data sizes:

| Images | Epochs | Time (CPU) | Accuracy |
|--------|--------|-----------|----------|
| 50 | 10 | 5 min | 60-70% |
| 100 | 15 | 15 min | 75-85% |
| 200 | 15 | 30 min | 85-92% |
| 500+ | 15 | 60 min | 90%+ |

---

## Using GPU (Optional, Faster)

If you have NVIDIA GPU:

```powershell
# Install GPU version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then train (will auto-use GPU)
python train_cnn.py
```

Training will be **10x faster** on GPU!

---

## Next Steps

1. ✅ Prepare your data
2. ✅ Run `python train_cnn.py`
3. ✅ Run `streamlit run app.py`
4. ✅ Test with your meme images
5. ✅ Push to GitHub with trained model? (optional, models are large)

---

## Course Project Checklist

- ✅ Deep Learning Model: CNN with ResNet50
- ✅ Training: Supervised learning on labeled data
- ✅ Evaluation: Validation accuracy during training
- ✅ Deployment: Streamlit web interface
- ✅ Input: Upload meme image
- ✅ Output: Classification + confidence + rating
- ✅ Documentation: This guide + code comments

**Ready for submission!** 🚀

---

**Questions?** Check the main README.md

