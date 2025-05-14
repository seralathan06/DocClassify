import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#load model
model = load_model('best_model.h5')

# Map class indices (ensure the same order used during training)
class_indices = training_set.class_indices
labels = dict((v, k) for k, v in class_indices.items())  # Invert dict to map index → label

def classify_document(img_path):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale like during training

        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index]

        # Output
        predicted_label = labels[class_index]
        print(f"Predicted Document Type: {predicted_label}")
        print(f"Confidence: {confidence:.2%}")

    except Exception as e:
        print("Error:", e)

# Example usage
user_input = input("Enter the path to the document image: ")
classify_document(user_input)
