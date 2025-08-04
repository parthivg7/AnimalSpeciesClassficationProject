import tensorflow as tf

model = tf.keras.models.load_model('animal_species_model1.h5')

from tensorflow.keras.preprocessing import image
import numpy as np


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Assuming you have a new image named 'new_animal.jpg'
new_image_path = 'H.jpeg'
prepped_image = prepare_image(new_image_path)

# Get the model's prediction
predictions = model.predict(prepped_image)

class_names = ['dog', 'horse', 'elephant', 'butterfly', 'hen','cat','bull','goat','spider','swirrel']  # Example

# Find the index with the highest probability
predicted_class_index = np.argmax(predictions)

# Get the corresponding class name
predicted_class = class_names[predicted_class_index]

print(f"The predicted species is: {predicted_class}")
print(f"Prediction probabilities: {predictions}")