import os
import opendatasets as od
od.download("https://www.kaggle.com/datasets/ritvik1909/document-classification-dataset")


import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

# Set path to dataset
dataset_path = 'document-classification-dataset'

# ImageDataGenerator with training-validation split
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)  # 20% for validation

# Load training set
training_set = train_datagen.flow_from_directory(dataset_path,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 subset='training',
                                                 shuffle=True)

# Load validation set
validation_set = train_datagen.flow_from_directory(dataset_path,
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   subset='validation',
                                                   shuffle=False)

# Invert class indices
class_indices = training_set.class_indices
labels = dict((v, k) for k, v in class_indices.items())

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(training_set.classes),
    y=training_set.classes
)

class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_indices), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model using class weights
history = model.fit(training_set,
                    steps_per_epoch=len(training_set),
                    epochs=25,
                    validation_data=validation_set,
                    validation_steps=len(validation_set),
                    callbacks=[early_stopping, model_checkpoint],
                    class_weight=class_weights_dict)

# Evaluate the model
loss, accuracy = model.evaluate(validation_set)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
