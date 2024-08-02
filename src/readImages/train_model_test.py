import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Function to load and preprocess test images
def preprocess_test_images(image_paths, target_size=(256, 256)):
    images = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            # Read and preprocess the image
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = resize_and_pad(image, target_size)  # Resize and pad to target size
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            images.append(image_array)
        else:
            print(f"Image file {image_path} not found.")
    return np.array(images)

# Function to resize and pad images while maintaining aspect ratio
def resize_and_pad(image, target_size):
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    new_image = Image.new('L', target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    return new_image

# Load the spreadsheet
spreadsheet_path = 'Data/PatientsTestingData.xlsx'
spreadsheet = pd.read_excel(spreadsheet_path)

# Extract file paths from the spreadsheet
test_image_paths = spreadsheet['Patient X-Ray File'].tolist()
#print(f"image path: {test_image_paths}%")

# Preprocess test images
test_images = preprocess_test_images(test_image_paths)
# Reshape images to add a channel dimension (for grayscale images)
test_images = test_images.reshape(-1, 256, 256, 1)

# Load the trained model
model = load_model('best_pneumonia_detection_model.keras')

# Make predictions
predictions = model.predict(test_images)
# Get the predicted classes (index of the highest probability)
predicted_classes = np.argmax(predictions, axis=1)

# Print out predictions
for i, image_path in enumerate(test_image_paths):
    print(f"Image: {image_path}, Predicted Class: {predicted_classes[i]}")

true_labels = spreadsheet['Pneumonia'].values  
# label_mapping = {'Y2': 2, 'Y1': 1, 'N': 0}
# spreadsheet['Pneumonia'] = spreadsheet['Pneumonia'].map(label_mapping)
print(f"\ntrue outcome {spreadsheet['Pneumonia']}")



# Calculate accuracy
accuracy = np.mean(predicted_classes == true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate and print other metrics like precision, recall, F1-score, etc.
#print(classification_report(true_labels, predicted_classes, target_names=['Class 0', 'Class 1', 'Class 2']))
