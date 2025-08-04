# 1. Unzip Dataset (if not already unzipped)
# You can do this manually or use Python's built-in zipfile module
import zipfile
import os

zip_file_path = 'animals10.zip'
extracted_dir_path = 'animals10'

# if not os.path.exists(extracted_dir_path):
#     print("Unzipping dataset...")
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(extracted_dir_path)
#     print("Unzipping complete.")

# 2. Set Up Paths and Preprocess Data



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

data_dir = os.path.join(extracted_dir_path, 'raw-img')
img_height, img_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

# 3. Load Pretrained Model and Customize
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Train the Model
print("\nTraining the model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
print("Model training complete.")

# 5. Visualize Accuracy and Loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# 6. Save the Model
model.save("animal_species_model1.h5")
print("Model saved as animal_species_model.h5")

# # 7. Predict on New Image
# def predict_species(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     class_idx = np.argmax(prediction)
#     class_label = list(train_data.class_indices.keys())[class_idx]
#     return class_label

# # To predict, you'll need to specify the path to a new image.
# # Example: replace 'path/to/your/image.jpg' with the actual path.
# new_image_path = 'path/to/your/image.jpg'  # <-- CHANGE THIS
# print("Predicting species for:", new_image_path)
# print("Predicted Species:", predict_species(new_image_path))