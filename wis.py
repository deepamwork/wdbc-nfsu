import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. Load the dataset using NumPy
# Each line: ID, Diagnosis (M/B), 30 numeric features
data = np.genfromtxt('wdbc.data', delimiter=',', dtype=str)

# 2. Prepare data
# Extract features (columns 2–31)
X = data[:, 2:].astype(float)

# Convert diagnosis: 'M' → 1, 'B' → 0
y = np.where(data[:, 1] == 'M', 1, 0).astype(float)

# 3. Build the ANN model
model = Sequential()

# Input + first hidden layer
# - 16 neurons: enough capacity for 30 features
# - input_dim=30: number of input features
# - activation='relu': helps learn complex relationships
model.add(Dense(units=16, activation='relu', input_dim=30))

# Second hidden layer
# - 8 neurons: smaller layer to refine learned features
# - activation='relu': keeps nonlinearity
model.add(Dense(units=8, activation='relu'))

# Output layer
# - 1 neuron: binary output (malignant or benign)
# - activation='sigmoid': converts to probability (0–1)
model.add(Dense(units=1, activation='sigmoid'))

# 4. Compile the model
# - adam: adaptive optimizer
# - binary_crossentropy: loss function for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train the model
# - epochs=100: how many times to go through all data
# - batch_size=10: number of samples per training update
model.fit(X, y, epochs=100, batch_size=10, verbose=1)

# 6. Predict & Evaluate
predictions = (model.predict(X) > 0.5).astype(int)
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
print("\nSample Predictions:", predictions[:10].ravel())
