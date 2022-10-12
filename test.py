import tensorflow as tf
import numpy as np

# Generate XOR training dataset
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
y_train = np.array([[0], [1], [1], [0]], "float32")

# Generate XOR test dataset
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
y_test = np.array([[0], [1], [1], [0]], "float32")

# Create a model with two Relu hidden layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1000, verbose=0)

# Evaluate the model
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Make predictions
predictions = model.predict(x_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

