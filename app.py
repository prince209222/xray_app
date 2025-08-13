import os
import streamlit as st
import torch
import torchvision
from torch import nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Configuration
MODEL_DIR = 'models'
MODEL_NAME = 'medical_resnet34.pt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# 1. Model definition (identical to training)
def medical_resnet34(num_classes=4, input_channels=1):
    model = torchvision.models.resnet34(pretrained=False)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    model.fc = nn.Linear(512, num_classes)
    return model

# 2. Model loading with enhanced error handling
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model = medical_resnet34(num_classes=4, input_channels=1)
        
        # Handle both full checkpoint and state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Verify class order matches training
        # class_names = ['normal', 'viral_pneumonia', 'lung_opacity', 'covid']  # UPDATE THIS
        class_names = ['covid','lung_opacity','normal','viral_pneumonia']
        return model, class_names
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# 3. Fixed preprocessing with proper normalization display
def get_transforms():
    def force_grayscale(image, **kwargs):
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        return image

    return A.Compose([
        A.Lambda(image=force_grayscale),
        A.Resize(256, 256),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])

def normalize_for_display(tensor):
    """Convert normalized tensor back to displayable image"""
    # Reverse normalization: (tensor * std) + mean
    image = tensor.numpy().transpose(1, 2, 0)
    image = (image * 0.229) + 0.485  # Reverse ImageNet normalization
    image = np.clip(image, 0, 1)  # Ensure valid range
    return (image * 255).astype(np.uint8)

def predict(image, model, transform):
    try:
        image_np = np.array(image)
        transformed = transform(image=image_np)
        tensor = transformed['image'].unsqueeze(0)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
            # Debug prints
            print("Raw logits:", logits.cpu().numpy())
            print("Probabilities:", probs.cpu().numpy())
            
            return probs.cpu().numpy()
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="X-ray Classifier",
        page_icon="🩻",
        layout="wide"
    )
    
    st.title("Medical X-ray Diagnosis")
    
    # Load model
    model, class_names = load_model()
    transform = get_transforms()
    
    # File upload
    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            cols = st.columns([1, 2])
            
            with cols[0]:
                st.image(img, caption="Original Image", use_container_width=True)
                
                # Show preprocessing steps
                with st.expander("Preprocessing Steps"):
                    # Convert to grayscale
                    gray = np.array(img.convert('L'))
                    st.image(gray, caption="Grayscale", width=256)
                    
                    # Show normalized version (properly scaled for display)
                    transformed = transform(image=np.array(img))
                    display_img = normalize_for_display(transformed['image'])
                    st.image(display_img, caption="Normalized", width=256)
            
            with cols[1]:
                with st.spinner("Analyzing..."):
                    probs = predict(img, model, transform)
                    
                    if probs is not None:
                        results = list(zip(class_names, probs))
                        results.sort(key=lambda x: x[1], reverse=True)
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['#4CAF50', '#FFC107', '#FF5722', '#E91E63']
                        
                        for i, (name, prob) in enumerate(results):
                            ax.barh(
                                name.replace('_', ' ').title(),
                                prob * 100,
                                color=colors[i],
                                alpha=0.6
                            )
                            ax.text(
                                prob * 100 + 1,
                                i,
                                f"{prob * 100:.1f}%",
                                va='center'
                            )
                        
                        ax.set_xlim(0, 100)
                        ax.set_title("Diagnosis Confidence")
                        st.pyplot(fig)
                        
                        # Show top prediction
                        top_class, top_prob = results[0]
                        st.success(f"Most likely: {top_class.replace('_', ' ').title()} ({top_prob * 100:.1f}%)")
                        
                        # Debug output
                        with st.expander("Raw Probabilities"):
                            st.write({name: f"{prob:.4f}" for name, prob in results})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()