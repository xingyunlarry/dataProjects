# this is the code for NN training using x-ray images from NIH. They are all in 1024x1024 resolution
# Run the following pip to install libraries in the IDE (Integrated Development Environment)
# pip install --upgrade pip 
# pip install sqlite3
# pip install imageio
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install openpyxl

import os # this the library to find files from the computer where the code is running
import imageio # this is the library to read images
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model #make sure to 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split # this is the library for python mechine learning
from PIL import Image

# Directory containing the images
DIR = "images"

# Load the spreadsheet
file_path = 'Data/PatientsTrainingData.xlsx'
spreadsheet = pd.read_excel(file_path)

# Map categorical labels to numeric values
#label_mapping = {'Y2': 2, 'Y1': 1, 'N': 0}
#spreadsheet['Pneumonia'] = spreadsheet['Pneumonia'].map(label_mapping)

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
            #print(f"Image file {image_path} found.")
        else:
            print(f"Image file {image_path} not found.")

    return np.array(images), np.array(pneumonias)

def pad_image_to_target(image, target_size):
    # Calculate the new size while maintaining aspect ratio
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new('L', target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    return new_image

# Example usage
# Assuming 'spreadsheet' is a DataFrame with the relevant columns
images, pneumonias = preprocess_data(spreadsheet, DIR, num_records=5200, target_size=(256, 256))

# Reshape images to add a channel dimension (for grayscale images)
images = images.reshape(-1, 256, 256, 1)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(images, pneumonias, test_size=0.15, random_state=42, stratify=pneumonias)

# Print shapes and types for debugging
print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
print("y_train shape:", y_train.shape, "dtype:", y_train.dtype)
print("X_test shape:", X_test.shape, "dtype:", X_test.dtype)
print("y_test shape:", y_test.shape, "dtype:", y_test.dtype)

# Define the model
image_input = Input(shape=(256, 256, 1), name='image')
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=image_input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=1,
    validation_data=(X_test, y_test)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Save the model
model.save('pneumonia_detection_model.h5')
print("Model saved as 'pneumonia_detection_model.h5'")


