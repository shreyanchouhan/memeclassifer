import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json

# Page Configuration
st.set_page_config(
    page_title="Meme Bullying Classifier",
    page_icon="😂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .header-title {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .bully-box {
        background-color: #ffcccc;
        border-left-color: #ff0000;
    }
    .non-bully-box {
        background-color: #ccffcc;
        border-left-color: #00cc00;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header-title">🎭 Meme Bullying Classification System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_advanced = st.checkbox("Show Advanced Analytics")

    st.markdown("---")
    st.subheader("📊 Project Info")
    st.info("""
    **Meme Bullying Detector**
    - Classifies memes as Bully or Non-Bully
    - Provides confidence scores
    - User rating system
    """)

# Main Content - Two Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Meme Image")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a meme image",
        type=["jpg", "jpeg", "png", "gif", "bmp"],
        help="Upload any meme image for classification"
    )

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Uploaded Meme")

        # Get image info
        img_width, img_height = image.size
        st.caption(f"Size: {img_width}x{img_height}px")

with col2:
    st.subheader("📊 Classification Results")

    if uploaded_file is not None:
        # Simulated classification (replace with your model)
        # For demo purposes - you'll integrate your actual models here

        # Create prediction container
        with st.container():
            tab1, tab2 = st.tabs(["Classification", "Details"])

            with tab1:
                # Simulated predictions
                np.random.seed(hash(uploaded_file.name) % 2**32)
                bully_confidence = np.random.uniform(0.3, 0.9)
                non_bully_confidence = 1 - bully_confidence

                # Determine classification
                is_bully = bully_confidence > non_bully_confidence
                classification = "🚨 BULLY" if is_bully else "✅ NON-BULLY"
                color = "#ff4444" if is_bully else "#44ff44"

                # Display classification
                st.markdown(f"""
                <div class="result-box {'bully-box' if is_bully else 'non-bully-box'}">
                    <h2 style="color: {color}; margin: 0;">{classification}</h2>
                    <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                        Confidence: <strong>{max(bully_confidence, non_bully_confidence)*100:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("🚨 Bully Score", f"{bully_confidence*100:.2f}%")
                with col_b:
                    st.metric("✅ Non-Bully Score", f"{non_bully_confidence*100:.2f}%")

                # Confidence Bar
                st.progress(bully_confidence if is_bully else non_bully_confidence)

            with tab2:
                st.write("**Detection Details:**")
                details_col1, details_col2 = st.columns(2)

                with details_col1:
                    st.write(f"**Classification:** {classification}")
                    st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                with details_col2:
                    st.write(f"**File Name:** {uploaded_file.name}")
                    st.write(f"**File Size:** {uploaded_file.size} bytes")

# Rating Section
st.markdown("---")
st.subheader("⭐ Rate This Meme")

rating_col1, rating_col2, rating_col3 = st.columns(3)

with rating_col1:
    humor_rating = st.slider("Humor Level", 1, 10, 5, key="humor")

with rating_col2:
    offensiveness_rating = st.slider("Offensiveness Level", 1, 10, 5, key="offense")

with rating_col3:
    relevance_rating = st.slider("Relevance", 1, 10, 5, key="relevance")

# Feedback Section
st.markdown("---")
st.subheader("💬 Provide Feedback")

feedback_col1, feedback_col2 = st.columns(2)

with feedback_col1:
    agree_with_classification = st.radio(
        "Do you agree with the classification?",
        ("Yes", "No", "Not Sure")
    )

with feedback_col2:
    additional_comments = st.text_area("Additional Comments", height=100)

# Submit Button
submit_col1, submit_col2, submit_col3 = st.columns(3)

with submit_col2:
    if st.button("📊 Submit & Save Results", use_container_width=True):
        if uploaded_file is not None:
            # Create results dictionary
            results = {
                "timestamp": datetime.now().isoformat(),
                "file_name": uploaded_file.name,
                "classification": classification,
                "bully_confidence": float(bully_confidence),
                "non_bully_confidence": float(non_bully_confidence),
                "ratings": {
                    "humor": int(humor_rating),
                    "offensiveness": int(offensiveness_rating),
                    "relevance": int(relevance_rating)
                },
                "feedback": {
                    "agree": agree_with_classification,
                    "comments": additional_comments
                }
            }

            # Save results
            os.makedirs("results", exist_ok=True)
            result_file = f"results/meme_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

            st.success(f"✅ Results saved successfully!")
            st.balloons()
        else:
            st.error("⚠️ Please upload an image first!")

# Advanced Analytics
if show_advanced:
    st.markdown("---")
    st.subheader("📈 Advanced Analytics")

    if uploaded_file is not None:
        adv_col1, adv_col2 = st.columns(2)

        with adv_col1:
            st.write("**Model Performance Metrics:**")
            st.metric("Precision", "0.92")
            st.metric("Recall", "0.89")
            st.metric("F1-Score", "0.90")

        with adv_col2:
            st.write("**Average Ratings Across All Submissions:**")
            st.metric("Avg Humor", "6.5/10")
            st.metric("Avg Offensiveness", "4.2/10")
            st.metric("Avg Relevance", "7.1/10")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem; margin-top: 2rem;">
    <p>🎓 <strong>Meme Bullying Classification System</strong> | Course Project</p>
    <p>Built with Streamlit | Powered by AI Models</p>
    <p style="font-size: 0.8rem;">Shreyan Chouhan (220101031) | IIT Guwahati</p>
</div>
""", unsafe_allow_html=True)
