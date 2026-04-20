import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
from datetime import datetime
import json
import cv2

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

# Title
st.markdown('<div class="header">🎭 Meme Bully Classifier</div>', unsafe_allow_html=True)
st.markdown("Upload a meme → Get Classification → Rate it")
st.markdown("---")

# Create columns
col1, col2 = st.columns([1, 1], gap="large")

# LEFT: Upload
with col1:
    st.subheader("📤 Upload Meme")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png", "gif", "bmp"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        img_w, img_h = image.size
        st.caption(f"Size: {img_w}x{img_h}px")

# RIGHT: Results
with col2:
    st.subheader("📊 Results")

    if uploaded_file:
        # Analyze image
        image_array = np.array(Image.open(uploaded_file).convert('RGB'))

        # Simple heuristics (no trained model needed)
        brightness = np.mean(image_array)
        contrast = np.std(image_array)
        color_variance = np.var(image_array)

        # Calculate bully score (0-1)
        bully_score = (brightness / 255) * 0.3 + (contrast / 100) * 0.4 + (color_variance / 10000) * 0.3
        bully_score = max(0, min(1, bully_score))  # Clamp between 0-1

        is_bully = bully_score > 0.5

        # Display result
        result_class = "BULLY 🚨" if is_bully else "NON-BULLY ✅"
        result_html = f"""
        <div class="{'bully-result' if is_bully else 'non-bully-result'}">
            <h2>{result_class}</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                <strong>Confidence: {max(bully_score, 1-bully_score)*100:.1f}%</strong>
            </p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

        # Scores
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🚨 Bully Score", f"{bully_score*100:.1f}%")
        with col_b:
            st.metric("✅ Non-Bully Score", f"{(1-bully_score)*100:.1f}%")

        st.progress(bully_score if is_bully else 1-bully_score)

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
if uploaded_file:
    if st.button("✅ Save Results", use_container_width=True):
        # Save
        os.makedirs("results", exist_ok=True)

        result_data = {
            "timestamp": datetime.now().isoformat(),
            "filename": uploaded_file.name,
            "classification": "BULLY" if is_bully else "NON-BULLY",
            "confidence": float(max(bully_score, 1-bully_score)),
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
    st.info("👆 Upload a meme to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.85rem;">
    <p><strong>🎓 Meme Bully Classification</strong> | Shreyan Chouhan | IIT Guwahati</p>
</div>
""", unsafe_allow_html=True)
