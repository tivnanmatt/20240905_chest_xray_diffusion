import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess the image
image = Image.open("../../data/NIH_Chest_Xray/images_001/images/00000009_000.png")

# Define the types of brain cancer
pneumonia_options = [
    "Chest Radiograph, Diagnosis: Pneumonia",
    "Chest Radiograph, Diagnosis: No Pneumonia"
]

# Prepare the inputs
inputs = processor(text=pneumonia_options, images=image,
                   return_tensors="pt", padding=True)

# Forward pass to get logits
outputs = model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image

# convert to probabilities
probs = logits_per_image.softmax(dim=1)

# Display the results
for i, diagnosis in enumerate(pneumonia_options):
    print(f"{diagnosis}: {probs[0][i].item() * 100:.2f}%")

# Find the predicted diagnosis
predicted_idx = probs.argmax(dim=1).item()
predicted_diagnosis = pneumonia_options[predicted_idx]

print(f"\nPredicted Diagnosis: {predicted_diagnosis}")
