import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import EfficientNet_B3_Weights # Import weights
import torch.nn.functional as F
import streamlit as st

device = torch.device("cpu")

st.title("Fi-Classify")

def load_model():
    weights = EfficientNet_B3_Weights.DEFAULT # Define weights for the new model
    model = models.efficientnet_b3(weights=weights) # Load efficientnet_b3

    # Modify the classifier for 16 classes (must match training)
    num_classes = 16
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    try:
        model.load_state_dict(torch.load('/workspaces/Fi-Classify/python/efficientnet_b3_rvl_cdip_small_200.pth',map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("File Not found")
        return None

class_names = {'advertisement': 0, 'budget': 1, 'email': 2, 'file_folder': 3, 'form': 4, 'handwritten': 5, 'invoice': 6, 'letter': 7, 'memo': 8, 'news_article': 9, 'presentation': 10, 'questionnaire': 11, 'resume': 12, 'scientific_publication': 13, 'scientific_report': 14, 'specification': 15}

idx_to_class = {v: k for k, v in class_names.items()}

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # convert grayscale to 3-channel
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_p, top_class = probabilities.topk(3, dim=1)
        top_p = top_p.squeeze().tolist()
        top_class = top_class.squeeze().tolist()

        st.subheader("Top 3 Predicted Classes:")
        for i in range(len(top_class)):
            class_index = top_class[i]
            confidence = top_p[i]
            predicted_label = idx_to_class.get(class_index, f"Unknown Class {class_index}")
            st.write(f"{predicted_label}: {confidence:.4f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")