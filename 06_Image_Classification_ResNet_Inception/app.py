import os
import json
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Import the model builders from our main script
from train_models import build_resnet50, build_inceptionv3

# Constants
OUTPUT_DIR = "./outputs"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.json")
CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# App Setup and Styling
st.set_page_config(page_title="Deep Learning Image Classifier", page_icon="🖼️", layout="centered")

# Basic styling
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #2C3E50; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .sub-header { font-size: 1.2rem; color: #7F8C8D; text-align: center; margin-bottom: 30px;}
    .result-text { font-size: 1.5rem; font-weight: bold; color: #27AE60; text-align: center;}
    .prob-text { font-size: 1rem; color: #34495E; text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Image Classification App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image and the model will guess its category among the 10 CIFAR-10 classes!</div>', unsafe_allow_html=True)

@st.cache_resource
def load_best_model():
    """Reads results.json, identifies the best model, and loads it."""
    try:
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
            
        resnet_acc = results["resnet50"]["best_test_acc"]
        inception_acc = results["inceptionv3"]["best_test_acc"]
        
        if resnet_acc >= inception_acc:
            model_name = "ResNet50"
            model = build_resnet50(10)
            model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "resnet50.pth"), map_location=DEVICE))
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            acc = resnet_acc
        else:
            model_name = "InceptionV3"
            model = build_inceptionv3(10)
            model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "inceptionv3.pth"), map_location=DEVICE))
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            acc = inception_acc
            
        model.to(DEVICE)
        model.eval()
        return model, model_name, transform, acc
    except Exception as e:
        return None, None, None, str(e)


# Status checking
if not os.path.exists(RESULTS_FILE):
    st.warning("Training is not complete yet! Please run the training and ensure 'outputs/results.json' is generated before using this app.")
    st.stop()
    
# Load model
with st.spinner("Loading best performing model..."):
    model, model_name, transform, stat = load_best_model()

if model is None:
    st.error(f"Error loading model: {stat}")
    st.stop()
    
st.success(f"Successfully loaded **{model_name}** as the winning model (Accuracy: {stat:.2f}%).")

uploaded_file = st.file_uploader("Choose an image to classify...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.markdown("---")
    
    # Classify the image
    with st.spinner(f"Analyzing with {model_name}..."):
        try:
            # Preprocess image
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
            top_prob, top_class_idx = torch.max(probabilities, 0)
            top_prob = top_prob.item() * 100
            top_class = CIFAR10_CLASSES[top_class_idx.item()]
            
            # Display results
            st.markdown(f'<div class="result-text">Prediction: {top_class}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prob-text">Confidence: {top_prob:.2f}%</div>', unsafe_allow_html=True)
            
            # Show a progress bar for the confidence
            st.progress(float(top_prob / 100))
            
            with st.expander("Show all probability scores"):
                for i, prob_val in enumerate(probabilities):
                    st.write(f"**{CIFAR10_CLASSES[i]}:** {prob_val.item()*100:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
