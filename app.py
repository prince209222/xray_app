import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
MODEL_PATH = '/app/models/medical_resnet34.pt'  # Docker absolute path
CLASS_DISPLAY = {
    'normal': ('Normal', '#4CAF50'),
    'viral_pneumonia': ('Viral Pneumonia', '#FFC107'),
    'lung_opacity': ('Lung Opacity', '#FF5722'),
    'covid': ('COVID-19', '#E91E63')
}

@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.16.0', 'resnet34', pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def predict(image, model):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ])
    tensor = transform(image=np.array(image))['image'].unsqueeze(0)
    with torch.inference_mode():
        return torch.nn.functional.softmax(model(tensor), dim=1)[0]

def main():
    st.set_page_config(page_title="X-ray Classifier", page_icon="🩻", layout="wide")
    st.title("🩻 Medical X-ray Diagnosis")
    
    model = load_model()
    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('L')
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(img, caption="Uploaded X-ray")
        
        with col2:
            with st.spinner("Analyzing..."):
                probs = predict(img, model)
                
                # Display results
                fig, ax = plt.subplots(figsize=(8, 4))
                for i, (cls, (name, color)) in enumerate(CLASS_DISPLAY.items()):
                    ax.barh(name, probs[i].item() * 100, color=color)
                ax.set_xlim(0, 100)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
