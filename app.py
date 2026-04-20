import streamlit as st
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json
import torch
import torch.nn.functional as F
from torchvision import transforms

# Import CNN model
from cnn_model import MemeBullyCNN

# Page config
st.set_page_config(
    page_title="Meme Bully Classifier",
    page_icon="😂",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main { padding: 2rem; }
.header { font-size: 2.5rem; color: #1f77b4; font-weight: bold; }
.bully-result { background-color: #ffcccc; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #ff0000; }
.non-bully-result { background-color: #ccffcc; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #00cc00; }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    """Load the trained CNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MemeBullyCNN(num_classes=2, pretrained=False)

    # Try to load trained model
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            return model, device, True
        except Exception as e:
            st.warning(f"⚠️ Could not load trained model: {e}")
            return model, device, False
    else:
        return model, device, False


def classify_meme(image, model, device):
    """Classify a meme using the CNN model"""

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Convert PIL image to tensor
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Convert to numpy
    confidence = confidence.cpu().numpy()[0]
    predicted = predicted.cpu().numpy()[0]

    # Get both class probabilities
    probs = probabilities.cpu().numpy()[0]
    non_bully_prob = probs[0]
    bully_prob = probs[1]

    return predicted, confidence, non_bully_prob, bully_prob


# Title
st.markdown('<div class="header">🎭 Meme Bully Classifier</div>', unsafe_allow_html=True)
st.markdown("Upload a meme → Get Classification → Rate it")
st.markdown("---")

# Load model
model, device, model_loaded = load_model()

if not model_loaded:
    st.warning("⚠️ No trained model found! Train the model first:")
    st.code("python train_cnn.py", language="bash")
    st.info("""
    Steps to train:
    1. Add images to: data/bully/ and data/non_bully/
    2. Run: python train_cnn.py
    3. Reload this page
    """)

# Create columns
col1, col2 = st.columns([1, 1], gap="large")

# LEFT: Upload
with col1:
    st.subheader("📤 Upload Meme")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png", "gif", "bmp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        img_w, img_h = image.size
        st.caption(f"Size: {img_w}x{img_h}px")

# RIGHT: Results
with col2:
    st.subheader("📊 Results")

    if uploaded_file and model_loaded:
        try:
            # Classify
            predicted, confidence, non_bully_prob, bully_prob = classify_meme(image, model, device)

            is_bully = predicted == 1

            # Display result
            result_class = "🚨 BULLY" if is_bully else "✅ NON-BULLY"
            result_html = f"""
            <div class="{'bully-result' if is_bully else 'non-bully-result'}">
                <h2>{result_class}</h2>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                    <strong>Confidence: {confidence*100:.1f}%</strong>
                </p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)

            # Scores
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🚨 Bully Score", f"{bully_prob*100:.1f}%")
            with col_b:
                st.metric("✅ Non-Bully Score", f"{non_bully_prob*100:.1f}%")

            st.progress(bully_prob if is_bully else non_bully_prob)

        except Exception as e:
            st.error(f"❌ Classification error: {e}")
    elif uploaded_file and not model_loaded:
        st.error("❌ Model not trained yet. Please train first.")
    elif uploaded_file:
        st.info("Waiting for image...")
    else:
        st.info("👆 Upload a meme to get started!")

# Ratings
st.markdown("---")
st.subheader("⭐ Rate This Meme")

rating_col1, rating_col2, rating_col3 = st.columns(3)
with rating_col1:
    humor = st.slider("Humor", 1, 10, 5)
with rating_col2:
    offense = st.slider("Offensiveness", 1, 10, 5)
with rating_col3:
    relevance = st.slider("Relevance", 1, 10, 5)

# Feedback
st.markdown("---")
st.subheader("💬 Feedback")

agree = st.radio("Do you agree?", ("Yes", "No", "Not Sure"))
comments = st.text_area("Comments", height=80)

# Submit
st.markdown("---")
if uploaded_file and model_loaded:
    if st.button("✅ Save Results", use_container_width=True):
        # Save
        os.makedirs("results", exist_ok=True)

        result_data = {
            "timestamp": datetime.now().isoformat(),
            "filename": uploaded_file.name,
            "classification": "BULLY" if is_bully else "NON-BULLY",
            "confidence": float(confidence),
            "bully_score": float(bully_prob),
            "non_bully_score": float(non_bully_prob),
            "ratings": {
                "humor": int(humor),
                "offensiveness": int(offense),
                "relevance": int(relevance)
            },
            "feedback": {
                "agree": agree,
                "comments": comments
            }
        }

        filename = f"results/meme_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)

        st.success("✅ Saved!")
        st.balloons()
else:
    if not model_loaded:
        st.warning("⚠️ Train the model first to save results")
    else:
        st.info("👆 Upload a meme to save results!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.85rem;">
    <p><strong>🎓 Meme Bully Classification System</strong></p>
    <p style="font-size: 0.8rem;">Team: Shreyan Chouhan, Dhruv Garg, Dibya | IIT Guwahati</p>
    <p style="font-size: 0.75rem;">Deep Learning Course Project | CNN-based Classification</p>
</div>
""", unsafe_allow_html=True)
