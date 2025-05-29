
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
