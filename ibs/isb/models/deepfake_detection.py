import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        output1 = self.model1(x)
        output2 = self.model2(x)
        avg_output = (output1 + output2) / 2
        return avg_output

def load_models(model1_path, model2_path):
    checkpoint1 = torch.load(model1_path, map_location=torch.device('cpu'))
    model1 = checkpoint1['model']
    model1.load_state_dict(checkpoint1['model_state_dict'])

    checkpoint2 = torch.load(model2_path, map_location=torch.device('cpu'))
    model2 = checkpoint2['model']
    model2.load_state_dict(checkpoint2['model_state_dict'])

    model1.eval()
    model2.eval()

    ensemble_model = EnsembleModel(model1, model2)
    return ensemble_model

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

def predict(image_path, model):
    input_tensor = preprocess_image(image_path)
    model.eval()

    with torch.no_grad():
        raw_prediction = model(input_tensor)
        probability = torch.sigmoid(raw_prediction)

    threshold = 0.1
    return "The image is likely a deepfake." if probability >= threshold else "The image is likely real."
