"""
Model Integration Module for Meme Bullying Classifier
Integrate your trained models here
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import easyocr
import pickle
import os


class MemeBullyingClassifier:
    """
    Main classifier that combines visual and textual features
    """

    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model_path = model_path

        # Initialize CLIP for visual understanding
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(device)
            print("✅ CLIP model loaded successfully")
        except Exception as e:
            print(f"⚠️ CLIP model loading failed: {e}")
            self.clip_model = None

        # Initialize OCR for text extraction
        try:
            self.ocr = easyocr.Reader(['en'])
            print("✅ OCR model loaded successfully")
        except Exception as e:
            print(f"⚠️ OCR model loading failed: {e}")
            self.ocr = None

        # Load trained classifier if available
        self.classifier = self._load_classifier(model_path)

    def _load_classifier(self, model_path):
        """Load pre-trained classifier"""
        if model_path and os.path.exists(model_path):
            try:
                classifier = pickle.load(open(model_path, 'rb'))
                print(f"✅ Classifier loaded from {model_path}")
                return classifier
            except Exception as e:
                print(f"⚠️ Failed to load classifier: {e}")
        return None

    def extract_visual_features(self, image):
        """
        Extract visual features using CLIP
        Returns: numpy array of features
        """
        if self.clip_model is None:
            return np.zeros(512)

        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting visual features: {e}")
            return np.zeros(512)

    def extract_text_features(self, image):
        """
        Extract text from image using OCR
        Returns: dict with text and text features
        """
        if self.ocr is None:
            return {"text": "", "text_length": 0}

        try:
            results = self.ocr.readtext(np.array(image))
            extracted_text = " ".join([text[1] for text in results])

            # Create simple text features
            text_features = {
                "text": extracted_text,
                "text_length": len(extracted_text),
                "word_count": len(extracted_text.split()),
                "has_caps": any(c.isupper() for c in extracted_text),
            }
            return text_features
        except Exception as e:
            print(f"Error extracting text: {e}")
            return {"text": "", "text_length": 0}

    def predict(self, image_path_or_pil):
        """
        Main prediction method
        Input: Image path (str) or PIL Image
        Output: dict with classification and confidence scores
        """
        # Load image
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')

        # Extract features
        visual_features = self.extract_visual_features(image)
        text_features = self.extract_text_features(image)

        # Combine features for classification
        # Simple heuristic-based classification (replace with your trained model)
        combined_features = self._combine_features(visual_features, text_features)

        # Predict
        if self.classifier is not None:
            try:
                prediction = self.classifier.predict(combined_features.reshape(1, -1))[0]
                confidence = self.classifier.predict_proba(combined_features.reshape(1, -1)).max()
            except Exception as e:
                print(f"Classifier prediction failed: {e}")
                prediction, confidence = self._fallback_prediction(combined_features)
        else:
            prediction, confidence = self._fallback_prediction(combined_features)

        # Format output
        result = {
            "classification": "Bully" if prediction == 1 else "Non-Bully",
            "is_bully": bool(prediction == 1),
            "bully_confidence": float(confidence) if prediction == 1 else float(1 - confidence),
            "non_bully_confidence": float(1 - confidence) if prediction == 1 else float(confidence),
            "extracted_text": text_features.get("text", ""),
            "text_length": text_features.get("text_length", 0),
        }

        return result

    def _combine_features(self, visual_features, text_features):
        """Combine visual and text features"""
        # Normalize visual features
        visual_norm = visual_features / (np.linalg.norm(visual_features) + 1e-8)

        # Create text feature vector
        text_vec = np.array([
            text_features.get("text_length", 0),
            text_features.get("word_count", 0),
            1.0 if text_features.get("has_caps", False) else 0.0,
        ])

        # Combine and pad
        combined = np.concatenate([visual_norm[:500], text_vec.flatten()])
        if len(combined) < 512:
            combined = np.pad(combined, (0, 512 - len(combined)), mode='constant')

        return combined[:512]

    def _fallback_prediction(self, features):
        """Fallback prediction if no model is trained"""
        # Use feature magnitude as a simple heuristic
        feature_sum = np.sum(np.abs(features[:100]))
        threshold = 50.0
        confidence = min(feature_sum / threshold, 1.0)
        prediction = 1 if feature_sum > threshold else 0
        return prediction, confidence


# Utility functions for batch processing
def classify_directory(directory_path, output_csv=None):
    """Classify all images in a directory"""
    classifier = MemeBullyingClassifier()
    results = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            image_path = os.path.join(directory_path, filename)
            try:
                result = classifier.predict(image_path)
                result['filename'] = filename
                results.append(result)
                print(f"✅ {filename}: {result['classification']}")
            except Exception as e:
                print(f"❌ {filename}: {e}")

    # Save results if requested
    if output_csv:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    return results


if __name__ == "__main__":
    # Test the classifier
    classifier = MemeBullyingClassifier()

    # Test with a sample image (uncomment if you have a test image)
    # result = classifier.predict("path/to/test/image.jpg")
    # print(result)

    print("🎭 Meme Bullying Classifier initialized successfully!")
