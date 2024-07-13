import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the ensemble model class
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        # Get predictions from both models
        output1 = self.model1(x)
        output2 = self.model2(x)

        # Average the predictions
        avg_output = (output1 + output2) / 2
        return avg_output

checkpoint1 = torch.load('/content/drive/MyDrive/model/CelebDF_model_20_epochs_99acc.pt',map_location=torch.device('cpu'))
model1 = checkpoint1['model']
model1.load_state_dict(checkpoint1['model_state_dict'])

checkpoint2 = torch.load('/content/drive/MyDrive/modelnew/FFPP.pt',map_location=torch.device('cpu'))
model2 = checkpoint2['model']
model2.load_state_dict(checkpoint2['model_state_dict'])

# Ensure both models are in evaluation mode
model1.eval()
model2.eval()






from lime import lime_image
import torch
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import cv2

from google.colab import drive
drive.mount('/content/drive')

checkpoint = torch.load('/content/drive/MyDrive/modelnew/CelebDF.pt')
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image_path = '/content/drive/MyDrive/dataset/fake/id26_id9_0000_7 - Copy - Copy - Copy.jpg'
image = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

model.eval()
def predict_fn(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
    return output

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((128, 128)),

    ])

    return transf

def get_preprocess_transform():
    transf = transforms.Compose([
        transforms.ToTensor()
    ])

    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()

from PIL import Image

# Convert the numpy array to a PIL image
image_pil = Image.fromarray(image)

# Pass the PIL image to the pill_transf function
test_pred = batch_predict([pill_transf(image_pil)])

def get_pil_transform():
    transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
    ])

    return transf

pill_transf = get_pil_transform()

# Pass the NumPy array directly to the pill_transf function
test_pred = batch_predict([pill_transf(image)])

test_pred

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(image)),
                                         batch_predict,
                                         top_labels=2,
                                         hide_color=0,
                                         num_samples=1000,)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5   , hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=6, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)

ensemble_model = EnsembleModel(model1, model2)

ensemble_model_path = '/content/drive/MyDrive/model/ensemble_model.pt'
torch.save(ensemble_model, ensemble_model_path)

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Load and preprocess the image
image_path = '/content/drive/MyDrive/test/real/real_225.jpg'  # Replace with your actual image path
input_tensor = preprocess_image(image_path)

ensemble_model.eval()

# Get the ensemble prediction
with torch.no_grad():
    raw_prediction = ensemble_model(input_tensor)
    probability = torch.sigmoid(raw_prediction)

print(f"Raw output: {raw_prediction}")
print(f"Probability: {probability}")

threshold = 0.1
if probability >= threshold:
    print("The input is likely a deepfake.")
else:
    print("The input is likely real.")
