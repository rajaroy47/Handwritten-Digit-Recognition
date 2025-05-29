import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.0
x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(x_train, y_train,
                    epochs=15,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop])

model.save('mnist_cnn.h5')
print("Model trained and saved.")
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.callbacks import EarlyStopping

# # Load and preprocess data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# # Build model
# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Flatten(),
#     layers.Dropout(0.5),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(10, activation='softmax')
# ])

# # Compile
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train
# early_stop = EarlyStopping(monitor='val_loss', patience=3)
# model.fit(x_train, y_train,
#           epochs=15,
#           validation_data=(x_test, y_test),
#           callbacks=[early_stop])

# # Save model
# model.save('mnist_cnn.h5')
