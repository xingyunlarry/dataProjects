import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory containing the images
DIR = "images"

# List of image files
image_files = [
    "00000011_000.png",
    "00000011_001.png",
    "00000011_002.png",
    "00000011_003.png",
    "00000011_004.png",
    "00000011_005.png",
    "00000011_006.png"
]

# Function to load and preprocess images
def load_and_preprocess_images(image_files, dir_path):
    images = []
    for file in image_files:
        image_path = os.path.join(dir_path, file)
        image = Image.open(image_path)
        image = image.resize((256, 256))
        image_array = np.array(image)
        images.append(image_array)
    return np.array(images)

# Load and preprocess images
arrays = load_and_preprocess_images(image_files, DIR)

# Normalize the arrays to the range [0, 1]
arrays = arrays / 255.0

# Create a boolean output array (7 x 1)
outputs = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int32)

# Reshape arrays to add a channel dimension (for grayscale images)
arrays = arrays.reshape(-1, 256, 256, 1)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(arrays, outputs, test_size=1/3, random_state=42, stratify=outputs)

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
model.save('simple_model.h5')
print("Model saved as 'simple_model.h5'")
