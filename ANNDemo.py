import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. Prepare the Data
# Example: XOR gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
y = np.array([[0], [1], [1], [0]])            # Target labels

# 2. Build the ANN Model
model = Sequential()

# Input layer and first hidden layer
# 'units' defines the number of neurons in the layer
# 'input_dim' is the number of input features
# 'activation' specifies the activation function (e.g., 'relu' for Rectified Linear Unit)
model.add(Dense(units=4, input_dim=2, activation='relu'))

# Output layer
# 'units=1' for binary classification
# 'activation='sigmoid'' for binary classification (outputs a probability between 0 and 1)
model.add(Dense(units=1, activation='sigmoid'))

# 3. Compile the Model
# 'optimizer' is the algorithm used to update weights (e.g., 'adam')
# 'loss' is the function to minimize during training (e.g., 'binary_crossentropy' for binary classification)
# 'metrics' are used to evaluate the model's performance (e.g., 'accuracy')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the Model
# 'epochs' is the number of times the entire dataset is passed forward and backward through the network
# 'batch_size' is the number of samples per gradient update
model.fit(X, y, epochs=1000, batch_size=1, verbose=0) # verbose=0 suppresses training output

# 5. Make Predictions
predictions = model.predict(X)
print("Predictions (probabilities):")
print(predictions)

# Convert probabilities to binary class labels (0 or 1)
binary_predictions = (predictions > 0.5).astype(int)
print("\nPredictions (binary):")
print(binary_predictions)

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")