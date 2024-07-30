import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate six random integer arrays of shape 255x255
arrays = np.random.randint(0, 256, (6, 255, 255), dtype=np.uint8)

# Normalize the arrays to the range [0, 1]
arrays = arrays / 255.0

# Create a boolean output array (6 x 1)
# Ensure at least 2 samples in each class
outputs = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)

# Reshape arrays to add a channel dimension (for grayscale images)
arrays = arrays.reshape(-1, 255, 255, 1)

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(arrays, outputs, test_size=1/3, random_state=42, stratify=outputs)

# Print shapes and types for debugging
print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
print("y_train shape:", y_train.shape, "dtype:", y_train.dtype)
print("X_test shape:", X_test.shape, "dtype:", X_test.dtype)
print("y_test shape:", y_test.shape, "dtype:", y_test.dtype)

# Define the model
image_input = Input(shape=(255, 255, 1), name='image')
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
