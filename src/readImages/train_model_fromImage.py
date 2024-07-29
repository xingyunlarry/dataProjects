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
from tensorflow.python.keras.models import Model #make sure to 
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split # this is the library for python mechine learning

# Directory containing the images
DIR = "images"

# Load the spreadsheet
file_path = 'Data/PatientsData.xlsx'
spreadsheet = pd.read_excel(file_path) 

# Map categorical labels to numeric values
label_mapping = {'Y': 1, 'N': 0}
spreadsheet['Pneumonia'] = spreadsheet['Pneumonia'].map(label_mapping)

# This the function of Preprocess data, this function is called after the function 
def preprocess_data(spreadsheet, dir_path, num_records=6): #this 6 is number of images for training in the folder. If more images, it can be changed
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
            xray_image = imageio.v3.imread(image_path)
            image_array = np.array(xray_image)
            images.append(image_array)
            pneumonias.append(pneumonia)
        else:
            print(f"Image file {xray_file} not found.")

    return np.array(images), np.array(pneumonias) #np.array(images) is an array of numbers from 0 to 255, which represent greyscales

# Get the records
images, pneumonias = preprocess_data(spreadsheet, DIR, num_records=6)

# Normalize images
images = images / 255.0

# Prepare input data
X_images = np.array(images, dtype=np.float32) # we need to change dtype from default float64 to float32 because tensorflow needs float32 for stable performance
y = np.array(pneumonias, dtype=np.int32)

# Reshape images to add a channel dimension (for grayscale images)
X_images = X_images.reshape(-1, X_images.shape[1], X_images.shape[2], 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42) #test_size amd random_state can be adjusted for traing performance

# Print shapes and types for debugging
print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
print("y_train shape:", y_train.shape, "dtype:", y_train.dtype)
print("X_test shape:", X_test.shape, "dtype:", X_test.dtype)
print("y_test shape:", y_test.shape, "dtype:", y_test.dtype)

# Define the model. This is the standard code for tensorflow. I guess we can adjust the numbers and activation for better fit
image_input = Input(shape=(X_images.shape[1], X_images.shape[2], 1), name='image')
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
print("Model saved as 'pneumonia_detection_model.h5'") #pneumonia_detection_model.h5 is an array 
