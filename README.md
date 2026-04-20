# 🎭 Meme Bullying Classification System

**Course Project: IIT Guwahati**  
**Team Members:**
- Shreyan Chouhan
- Dhruv Garg
- Dibya

---

## 📋 Project Overview

This project implements a **machine learning system** that classifies memes as **Bully** or **Non-Bully** and provides a rating system. It combines:

- 🖼️ **Visual Analysis** (CLIP - Contrastive Language-Image Pre-training)
- 📝 **Text Extraction** (EasyOCR)
- 🔬 **Custom Classification Model** (To be trained with your data)
- 🎨 **Web Interface** (Streamlit)

---

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Run the Web Application**

```bash
streamlit run meme_classifier_app.py
```

The app will open at: `http://localhost:8501`

---

## 📁 Project Structure

```
MemeClassifier_Project/
├── meme_classifier_app.py       # Main Streamlit web application
├── model_integration.py         # Model loading and prediction logic
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── models/                      # (Create this folder for trained models)
│   └── trained_model.pkl       # Your trained classifier
├── data/                        # (Create this folder for training data)
│   ├── bully/                  # Bully meme images
│   └── non_bully/              # Non-bully meme images
└── results/                     # (Auto-created) Classification results
    └── meme_YYYYMMDD_HHMMSS.json
```

---

## 🔧 How to Use

### **A. Running the Web Interface**

```bash
streamlit run meme_classifier_app.py
```

**Features:**
- ✅ Upload meme images (JPG, PNG, GIF, BMP)
- ✅ Get instant classification (Bully/Non-Bully)
- ✅ View confidence scores
- ✅ Rate memes (Humor, Offensiveness, Relevance)
- ✅ Provide feedback
- ✅ View advanced analytics

### **B. Training Your Model**

```python
from model_integration import MemeBullyingClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# Initialize classifier for feature extraction
classifier = MemeBullyingClassifier()

# Load your images and labels
# Extract features for all images
# Train your model
# Save it

pickle.dump(trained_model, open('models/trained_model.pkl', 'wb'))
```

### **C. Batch Processing Images**

```python
from model_integration import classify_directory

# Classify all images in a folder
results = classify_directory(
    directory_path='path/to/memes',
    output_csv='results/batch_results.csv'
)
```

---

## 📊 Classification Output

The system returns a JSON object with:

```json
{
  "classification": "Bully",
  "is_bully": true,
  "bully_confidence": 0.85,
  "non_bully_confidence": 0.15,
  "extracted_text": "Your meme contains...",
  "text_length": 25,
  "timestamp": "2024-04-20T10:30:00",
  "ratings": {
    "humor": 7,
    "offensiveness": 8,
    "relevance": 6
  }
}
```

---

## 🤖 Model Architecture

### **Visual Features**
- Uses **CLIP (ViT-B/32)** pre-trained model
- Extracts 512-dimensional image embeddings
- Captures semantic meaning of meme visuals

### **Text Features**
- **EasyOCR** for text extraction from images
- Text length, word count, capitalization patterns
- Semantic understanding of meme captions

### **Classification**
- Combines visual + textual features
- Custom trained classifier (Random Forest / SVM / Neural Network)
- Provides confidence scores for both classes

---

## 📈 Training with Your Data

### **Step 1: Organize Data**
```
data/
├── bully/
│   ├── meme1.jpg
│   ├── meme2.jpg
│   └── ...
└── non_bully/
    ├── meme1.jpg
    ├── meme2.jpg
    └── ...
```

### **Step 2: Create Training Script**

```python
import os
import numpy as np
from model_integration import MemeBullyingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize feature extractor
extractor = MemeBullyingClassifier()

# Load images and extract features
X = []
y = []

# Process bully images
for img_file in os.listdir('data/bully/'):
    img_path = os.path.join('data/bully/', img_file)
    features = extractor.extract_visual_features(img_path)
    X.append(features)
    y.append(1)  # Bully label

# Process non-bully images
for img_file in os.listdir('data/non_bully/'):
    img_path = os.path.join('data/non_bully/', img_file)
    features = extractor.extract_visual_features(img_path)
    X.append(features)
    y.append(0)  # Non-bully label

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_scaled, y)

# Save model
os.makedirs('models', exist_ok=True)
pickle.dump(clf, open('models/trained_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

print("✅ Model trained and saved!")
```

### **Step 3: Run Training**
```bash
python train_model.py
```

---

## 🎯 Performance Metrics

The system will track:
- **Precision**: How many detected bullies are actually bullies
- **Recall**: How many actual bullies are detected
- **F1-Score**: Balanced measure of precision and recall
- **Accuracy**: Overall correctness

---

## 📝 Example Usage

```python
from model_integration import MemeBullyingClassifier
from PIL import Image

# Initialize classifier
clf = MemeBullyingClassifier(model_path='models/trained_model.pkl')

# Predict on a single image
image = Image.open('meme.jpg')
result = clf.predict(image)

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['bully_confidence']:.2%}")
print(f"Extracted Text: {result['extracted_text']}")
```

---

## 📂 File Descriptions

| File | Purpose |
|------|---------|
| `meme_classifier_app.py` | Main Streamlit web interface |
| `model_integration.py` | Core classification logic |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation |

---

## 🔒 Important Notes

1. **Data Privacy**: All user-uploaded images are processed locally
2. **Model Training**: Use your downloaded dataset for training
3. **GPU Support**: Uncomment `device='cuda'` in classifier for GPU acceleration
4. **Model Saving**: Always save trained models in the `models/` folder

---

## 🐛 Troubleshooting

### Issue: "CLIP model not found"
**Solution**: Install transformers
```bash
pip install --upgrade transformers
```

### Issue: "OCR not working"
**Solution**: Install easyocr
```bash
pip install easyocr
```

### Issue: Streamlit won't start
**Solution**: Update streamlit
```bash
pip install --upgrade streamlit
```

---

## 🎓 Course Requirements Checklist

- ✅ Input meme image
- ✅ Classify as Bully/Non-Bully
- ✅ Show confidence scores
- ✅ Rate meme (Humor, Offensiveness, Relevance)
- ✅ Visual web interface
- ✅ Save results to file
- ✅ Model integration
- ✅ Feature extraction
- ✅ Classification system

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review your data format and paths
- Ensure all dependencies are installed correctly

---

## 📄 License

Academic Project - IIT Guwahati

**Created:** April 2026  
**Last Updated:** April 20, 2026

---

**Happy Meme Classifying! 🚀**
