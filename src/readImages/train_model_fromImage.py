import os  # this the library to find files from the computer where the code is running
import imageio  # this is the library to read images
import numpy as np  # this is the library for numerical operations
import pandas as pd  # this is the library for data manipulation
import tensorflow as tf  # this is the library for deep learning
from tensorflow.keras.models import Model  # used to define a Keras model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout  # used to define layers in the model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # used for data augmentation
from sklearn.model_selection import train_test_split  # this is the library for splitting data into train/test sets
from sklearn.preprocessing import LabelEncoder  # used to encode labels
from sklearn.metrics import classification_report  # used to generate a classification report
from PIL import Image  # this is the library to handle image operations

# Directory containing the images
DIR = "images"

# Load the spreadsheet
file_path = 'Data/PatientsTrainingData.xlsx'
spreadsheet = pd.read_excel(file_path)  # read the excel file into a pandas DataFrame

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
images = images.reshape(-1, 256, 256, 1)  # Reshape images to add a channel dimension (for grayscale images)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(pneumonias)  # Convert categorical labels to numeric values
num_classes = len(np.unique(encoded_labels))  # Get the number of unique classes

# Split data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(images, encoded_labels, test_size=0.15, random_state=42, stratify=encoded_labels)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Randomly rotate images in the range 0-10 degrees
    width_shift_range=0.1,  # Randomly translate images horizontally
    height_shift_range=0.1,  # Randomly translate images vertically
    zoom_range=0.1,  # Randomly zoom into images
    horizontal_flip=True  # Randomly flip images horizontally
)
datagen.fit(X_train)  # Compute the data augmentation on the training set

# Define the model
image_input = Input(shape=(256, 256, 1), name='image')  # Input layer
x = Conv2D(32, (3, 3), activation='relu')(image_input)  # Convolutional layer with 32 filters
x = MaxPooling2D((2, 2))(x)  # Max-pooling layer
x = Conv2D(64, (3, 3), activation='relu')(x)  # Convolutional layer with 64 filters
x = MaxPooling2D((2, 2))(x)  # Max-pooling layer
x = Flatten()(x)  # Flatten the tensor
x = Dense(128, activation='relu')(x)  # Fully connected layer with 128 units
x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
output = Dense(num_classes, activation='softmax')(x)  # Output layer with softmax activation

# Create the model
model = Model(inputs=image_input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define model checkpoint callback to save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_pneumonia_detection_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model with validation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # Train the model using data augmentation
    epochs=2,  # Train for 100 epochs
    validation_data=(X_val, y_val),  # Validate on the validation set
    callbacks=[checkpoint]  # Use the model checkpoint callback
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)  # Evaluate the model on the test set
print(f"\nTest accuracy: {test_acc}")

# Load the best model
best_model = tf.keras.models.load_model('best_pneumonia_detection_model.keras')

# Get predictions
predictions = best_model.predict(X_test)  # Make predictions on the test set
predicted_classes = np.argmax(predictions, axis=1)  # Get the class with the highest probability for each prediction

# Convert encoded labels back to original labels for classification report
target_names = label_encoder.inverse_transform(np.unique(encoded_labels)).astype(str)

# Print out predictions
for i in range(len(X_test)):
    print(f"Image {i}: Predicted Class: {predicted_classes[i]}")

# Calculate and print classification report
print(classification_report(y_test, predicted_classes, target_names=target_names))  # Print the classification report

# Save the best model
best_model.save('best_pneumonia_detection_model.keras')  # Save the best model to a file
print("Best model saved as 'best_pneumonia_detection_model.keras'")
