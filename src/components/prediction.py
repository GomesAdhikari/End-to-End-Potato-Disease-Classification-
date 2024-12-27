import tensorflow as tf
from PIL import Image
import pickle
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = r"C:\Users\gomes\OneDrive\ML Krish Naik\Potato disease classification CNN\artifacts\models\final_model.keras"
model = load_model(model_path)

# Load the class-to-index mapping
class_to_idx_path = r"C:\Users\gomes\OneDrive\ML Krish Naik\Potato disease classification CNN\artifacts\preprocessing\class_mapping.pkl"
with open(class_to_idx_path, "rb") as file:
    class_to_idx = pickle.load(file)

# Invert the class_to_idx mapping to get index-to-class
idx_to_class = {idx: name for name, idx in class_to_idx.items()}

def predict_image(model, img):
    # Preprocess the image (resize and convert to array)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = tf.argmax(predictions[0]).numpy()  # Convert Tensor to NumPy

    # Print the predicted class index and name
    predicted_class_name = idx_to_class[predicted_class_index]
    print(f"Predicted Class Index: {predicted_class_index}")
    print(f"Predicted Class Name: {predicted_class_name}")

# Provide the path to the image
img_path = r''

# Open and preprocess the image (resize to match model input)
img = Image.open(img_path).resize((128, 128))  # Resize to model's input size

# Predict the class of the image
predict_image(model, img)
