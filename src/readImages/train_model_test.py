import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Directory containing the images
DIR = "images"

# Load the spreadsheet
file_path = 'Data/PatientsData.xlsx'
spreadsheet = pd.read_excel(file_path)

# Map categorical labels to numeric values
label_mapping = {'Y': 1, 'N': 0}
spreadsheet['Pneumonia'] = spreadsheet['Pneumonia'].map(label_mapping)

def preprocess_data(spreadsheet, dir_path, num_records=6):
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
            image = image.resize((256, 256), Image.LANCZOS)  # Resize to 256x256
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            images.append(image_array)
            pneumonias.append(pneumonia)
        else:
            print(f"Image file {xray_file} not found.")

    return np.array(images), np.array(pneumonias)

# Get the records
images, pneumonias = preprocess_data(spreadsheet, DIR, num_records=6)

# Reshape images to add a channel dimension (for grayscale images)
images = images.reshape(-1, 256, 256, 1)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(images, pneumonias, test_size=1/3, random_state=42, stratify=pneumonias)

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
    batch_size=2,
    validation_data=(X_test, y_test)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Save the model
model.save('pneumonia_detection_model.h5')
print("Model saved as 'pneumonia_detection_model.h5'")
