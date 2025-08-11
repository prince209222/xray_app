import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- Version Check ---
import sys
if not (3, 9) <= sys.version_info < (3, 10):
    import streamlit as st
    st.error(f"❌ Python 3.9.x required (Detected: {sys.version})")
    st.stop()

# --- Pre-Flight Checks ---
MODEL_PATH = 'models/medical_resnet34.pt'
if not os.path.exists(MODEL_PATH):
    import streamlit as st
    st.error(f"❌ Model not found at: {os.path.abspath(MODEL_PATH)}")
    st.stop()

# --- Imports with Try-Catch ---
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

# --- Constants ---
CLASSES = ['viral_pneumonia', 'covid', 'lung_opacity', 'normal']  # Model's output order
DISPLAY_CONFIG = {
    'normal': {'name': 'Normal', 'color': '#4CAF50'},
    'viral_pneumonia': {'name': 'Viral Pneumonia', 'color': '#FFC107'},
    'lung_opacity': {'name': 'Lung Opacity', 'color': '#FF5722'},
    'covid': {'name': 'COVID-19', 'color': '#E91E63'}
}

# --- Model Loader ---
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('pytorch/vision:v0.16.0', 'resnet34', pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed. Technical details:\n{str(e)}")
        st.stop()

# --- Streamlit App ---
def main():
    st.set_page_config(
        page_title="X-ray Classifier",
        page_icon="🩻",
        layout="wide"
    )
    
    st.title("🩻 Medical X-ray Classifier")
    model = load_model()
    
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Upload a chest X-ray (JPEG/PNG)
        2. Wait for analysis (typically 3-5 seconds)
        3. Review results
        """)
    
    uploaded_file = st.file_uploader("Choose X-ray", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert('L')  # Force grayscale
            img_np = np.array(img)
            
            # Prediction pipeline
            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
            
            with torch.no_grad():
                tensor = transform(image=img_np)['image'].unsqueeze(0)
                probs = torch.nn.functional.softmax(model(tensor), dim=1)[0].numpy()
            
            # Display results
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, caption="Uploaded X-ray", use_column_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                for i, cls in enumerate(CLASSES):
                    ax.barh(
                        DISPLAY_CONFIG[cls]['name'],
                        probs[i] * 100,
                        color=DISPLAY_CONFIG[cls]['color']
                    )
                ax.set_xlim(0, 100)
                ax.bar_label(ax.containers[0], fmt='%.1f%%')
                st.pyplot(fig)
                
                # Top prediction
                max_idx = np.argmax(probs)
                st.success(f"""
                    **Diagnosis:**  
                    :{DISPLAY_CONFIG[CLASSES[max_idx]]['color']}[**{DISPLAY_CONFIG[CLASSES[max_idx]]['name']}**]  
                    **Confidence:** {probs[max_idx]*100:.1f}%
                """)
        
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")

if __name__ == "__main__":
    main()

