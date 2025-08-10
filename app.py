import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Prevent libiomp5 errors
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**64)  # Prevent decompression bombs


# --- Critical OpenCV Installation Check ---
try:
    import cv2
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.0.74"])
    import cv2  # Now guaranteed to work

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- App Config ---
st.set_page_config(
    page_title="Medical X-ray Classifier",
    page_icon="🩻",
    layout="wide"
)

# --- Title ---
st.title("🩻 Medical X-ray Disease Classifier")
st.markdown("""
Upload a chest X-ray image to detect:
- **Normal** 
- **Viral Pneumonia** 
- **Lung Opacity** 
- **COVID-19**
""")

# --- Transformation Pipeline ---
def get_xray_transforms(img_size=256):
    def force_grayscale(image, **kwargs):
        """Convert to 1-channel grayscale using OpenCV"""
        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        return image

    return A.Compose([
        A.Lambda(image=force_grayscale),
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Initialize model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    
    # Adjust first layer for 1-channel input
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    
    # Load trained weights
    checkpoint = torch.load(
        'models/medical_resnet34.pt',
        map_location=torch.device('cpu')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# --- Prediction Function ---
def predict(image):
    transform = get_xray_transforms()
    image_np = np.array(image)
    tensor = transform(image=image_np)['image'].unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
    
    return probs.numpy()

# --- Class Labels ---
CLASSES = ['Normal', 'Viral Pneumonia', 'Lung Opacity', 'COVID-19']
COLORS = ['#4CAF50', '#FFC107', '#FF5722', '#E91E63']  # Green, Amber, Deep Orange, Pink

# --- Load Model ---
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# --- Streamlit UI ---
uploaded_file = st.file_uploader(
    "**Upload X-ray Image**", 
    type=["png", "jpg", "jpeg"],
    help="Chest X-ray images only"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        with st.spinner("Analyzing..."):
            try:
                # Get predictions
                probabilities = predict(image)
                
                # Plot results
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(
                    CLASSES, 
                    probabilities * 100,
                    color=COLORS
                )
                
                # Formatting
                ax.set_xlim(0, 100)
                ax.set_xlabel('Confidence (%)')
                ax.set_title('Disease Probability')
                ax.bar_label(bars, fmt='%.1f%%', padding=5)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Highlight top prediction
                max_idx = np.argmax(probabilities)
                st.success(f"""
                    **Most likely:**  
                    :{COLORS[max_idx]}[**{CLASSES[max_idx]}**]  
                    Confidence: **{probabilities[max_idx]*100:.1f}%**
                """)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("""
*Note: This AI tool assists healthcare professionals but doesn't replace clinical diagnosis.  
Model trained on ResNet34 with Albumentations preprocessing.*
""")

# --- Hidden Imports for Deployment ---
try:
    import cv2
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.0.74"])
    import cv2
