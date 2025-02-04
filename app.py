import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
import logging
import os

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model.
logging.info("Loading the model...")
model_path = "model/model.pth"
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode.

# Define image preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path):
    """
    Function to preprocess the image, make a prediction, and return the class index.
    """
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    prediction = output.argmax(1).item()
    return prediction


@app.route('/predict', methods=['POST'])
def predict_image():
    """
    API endpoint for image prediction.
    Accepts an image file via POST request and returns the predicted class.
    """
    if 'image' not in request.files:
        logging.error("No image file found in request.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        logging.error("Empty file received.")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded image
        image_path = "uploaded_image.jpg"
        file.save(image_path)

        # Call the prediction function
        logging.info("Processing the uploaded image...")
        label = predict(image_path)

        # Remove the temporary image
        os.remove(image_path)
        logging.info(f"Prediction successful: {label}")

        return jsonify({"prediction": label})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction"}), 500


if __name__ == '__main__':
    # Start the Flask app
    logging.info("Starting the API server...")
    app.run(host='0.0.0.0', port=5000)