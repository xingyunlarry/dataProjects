import os
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from PIL import Image

# Directory containing the images
DIR = "images"

# Load the spreadsheet
file_path = 'Data/PatientsTrainingData.xlsx'
spreadsheet = pd.read_excel(file_path)

# Function to preprocess images
def preprocess_data(spreadsheet, dir_path, num_records=5200, target_size=(256, 256)):
    images = []
    pneumonias = []

    for index, row in spreadsheet.iterrows():
        if index >= num_records:
            break

        xray_file = row['Patient X-Ray File']
        pneumonia = row['Pneumonia']

        # Load the X-ray image
        image_path = os.path.join(dir_path, xray_file)
        if os.path.exists(image_path):
            # Read and preprocess the image
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = pad_image_to_target(image, target_size)  # Pad to target size
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            images.append(image_array)
            pneumonias.append(pneumonia)
        else:
            print(f"Image file {image_path} not found.")

    return np.array(images), np.array(pneumonias)

def pad_image_to_target(image, target_size):
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    new_image = Image.new('L', target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    return new_image

# Preprocess data
images, pneumonias = preprocess_data(spreadsheet, DIR, num_records=5200, target_size=(256, 256))
images = images.reshape(-1, 256, 256, 1)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(pneumonias)
num_classes = len(np.unique(encoded_labels))

# Split data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(images, encoded_labels, test_size=0.15, random_state=42, stratify=encoded_labels)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Define the model
image_input = Input(shape=(256, 256, 1), name='image')
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=image_input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_val, y_val)
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Get predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Print out predictions
for i in range(len(X_test)):
    print(f"Image {i}: Predicted Class: {predicted_classes[i]}")

# Calculate and print classification report
target_names = label_encoder.classes_
print(classification_report(y_test, predicted_classes, target_names=target_names))

# Save the model
model.save('pneumonia_detection_model.h5')
print("Model saved as 'pneumonia_detection_model.h5'")
