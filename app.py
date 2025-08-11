import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- Version Check ---
import sys
if not (3, 9) <= sys.version_info < (3, 10):
    import streamlit as st
    st.error(f"❌ Requires Python 3.9.x (Detected: {sys.version})")
    st.stop()

# --- Constants ---
MODEL_PATH = 'models/medical_resnet34.pt'
CLASSES = ['viral_pneumonia', 'covid', 'lung_opacity', 'normal']

# --- Pre-Flight Checks ---
if not os.path.exists(MODEL_PATH):
    import streamlit as st
    st.error(f"❌ Model not found at: {os.path.abspath(MODEL_PATH)}")
    st.stop()

# --- Imports ---
try:
    import cv2
    import streamlit as st
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as e:
    import streamlit as st
    st.error(f"❌ Missing dependency: {str(e)}\nRun: pip install -r requirements.txt")
    st.stop()

# --- Model Loader ---
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.16.0', 'resnet34', pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# --- Streamlit App ---
def main():
    st.set_page_config(
        page_title="Medical X-ray Classifier",
        page_icon="🩻",
        layout="wide"
    )
    
    st.title("🩻 X-ray Diagnosis")
    model = load_model()
    
    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            img = Image.open(uploaded_file).convert('L')
            st.image(img, caption="Uploaded X-ray")
        
        with col2:
            with st.spinner("Analyzing..."):
                # Prediction
                transform = A.Compose([
                    A.Resize(256, 256),
                    A.Normalize(mean=[0.485], std=[0.229]),
                    ToTensorV2()
                ])
                
                tensor = transform(image=np.array(img))['image'].unsqueeze(0)
                
                with torch.inference_mode():
                    probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]
                
                # Display
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(
                    ['Normal', 'Viral', 'Opacity', 'COVID'],
                    probs.numpy() * 100,
                    color=['#4CAF50', '#FFC107', '#FF5722', '#E91E63']
                )
                ax.set_xlim(0, 100)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
