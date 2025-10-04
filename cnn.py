import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# loading and preparing the dataset 
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data() # x_train the input images (handwritten numbers) y_train = lables and test being the images the model has never seen
#datasets.mnist.load_data() actaully getting tensor flow to get the info
# normalising the  pixel values - why do we normalise - well NN can train faster and more relaibly if inputs are in a small range 0-1 thefore we divide by 255.0
x_train, x_test = x_train / 255.0, x_test / 255.0

# adding channel dimension (needed for CNN: width, height, channels) - channel is one of three values (RGB) and  helps the NN interpret the images as grayscale images - because the NN expects images as 3D objects with give height width and channel
x_train = x_train[..., None]
x_test = x_test[..., None]


# building the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # prevents overfitting
    layers.Dense(10, activation='softmax')  # 10 classes: digits 0–9
])
# valadation data is the same as data the model hasnt seen before
# training accuracy = how well it memorises the training set
# validation accuracy = how well it generalises it - good training and vad valadation = overfitting

# compile model using gradient based methods like ADAM
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# optimiser = telling NN how to adjust its weights every step
# loss = sparse_categorical_crossentropy - how wrong the predictions are (a mathamatical formula comparing predicted vs the true labels)
# metrics (accuracy) =  what to report so we humans can track progress
# optimizer fixes mistakes, loss measures mistakes, metrics shows me accuracy


# training the model of the data set full of images from 0-9 from tensorflow

history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))
# when you train with model.fit tensorflow keeps history in order to rememeber training and validations per epoch so we can plot later

# evaluating the success of the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
# verbose is how much info keras outputs when training NN 2 is one line per epoch

# plot training vs validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# 7. Test on a single example

import numpy as np

# picking a random test image
idx = np.random.randint(0, len(x_test))
img = x_test[idx]

plt.imshow(img.squeeze(), cmap='gray')
plt.title("True Label: " + str(y_test[idx]))
plt.show()

# making the prediction
prediction = model.predict(img.reshape(1,28,28,1))
predicted_label = np.argmax(prediction)
print("The Model's Prediction:", predicted_label)


#tensorflow is the big libary that does all the maths and keras is a simple frontend that allows us to connect and build a NN quickly by providing a shortcut
