from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import pickle
from tensorflow.keras.models import load_model
import uuid

# Initialize Flask app
app = Flask(__name__)

# Load the model and class-to-index mapping
model_path = r"C:\Users\gomes\OneDrive\ML Krish Naik\Potato disease classification CNN\artifacts\models\final_model.keras"
model = load_model(model_path)

# Load the class-to-index mapping
class_to_idx_path = r"C:\Users\gomes\OneDrive\ML Krish Naik\Potato disease classification CNN\artifacts\preprocessing\class_mapping.pkl"
with open(class_to_idx_path, "rb") as file:
    class_to_idx = pickle.load(file)

# Invert the class_to_idx mapping to get index-to-class
idx_to_class = {idx: name for name, idx in class_to_idx.items()}

def convert_to_jpg(image):
    """
    Convert image to JPG format if it's PNG
    :param image: PIL Image object
    :return: PIL Image object in RGB mode
    """
    # Convert RGBA to RGB if image has alpha channel
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Create a white background image
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the image on the background if it has alpha channel
        if image.mode in ('RGBA', 'LA'):
            background.paste(image, mask=image.split()[-1])
        else:
            background.paste(image)
        return background
    elif image.mode != 'RGB':
        return image.convert('RGB')
    return image

def predict_image(img):
    """
    Predict the class of the image.
    :param img: Image to classify
    :return: Predicted class name
    """
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    
    # Get the predicted class name
    predicted_class_name = idx_to_class[predicted_class_index]
    return predicted_class_name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint that accepts an image and returns the prediction
    :return: Rendered template with prediction
    """
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    
    if file:
        try:
            # Generate a unique filename to avoid conflicts
            file_extension = os.path.splitext(file.filename)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            upload_path = os.path.join('uploads', unique_filename)
            
            # Save the uploaded file
            file.save(upload_path)
            
            # Open the image with PIL
            img = Image.open(upload_path)
            
            # Convert to JPG if it's a PNG
            img = convert_to_jpg(img)
            
            # Resize the image
            img = img.resize((128, 128))
            
            # If the original was a PNG, save the converted JPG version
            if file_extension.lower() == '.png':
                jpg_path = os.path.join('uploads', f"{uuid.uuid4()}.jpg")
                img.save(jpg_path, 'JPEG')
                # Remove the original PNG file
                os.remove(upload_path)
                upload_path = jpg_path
            
            # Get prediction
            predicted_class = predict_image(img)
            
            # Clean up - remove the uploaded file
            os.remove(upload_path)
            
            return render_template('index.html', predicted_class=predicted_class)
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return render_template('index.html', error=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    # Ensure the uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Run the app
    app.run(debug=True)