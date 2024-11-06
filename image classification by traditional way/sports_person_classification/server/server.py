# This Flask application sets up a server with an API endpoint at /classify_image to classify images of sports celebrities. When an image in base64 format is received as a POST request, it passes this image data to the classify_image function in the util module for classification. The server responds with the classification results in JSON format and enables Cross-Origin Resource Sharing (CORS) for all origins. Before starting, it loads necessary artifacts (like the model and class dictionaries) and runs on port 5000.

from flask import Flask, request, jsonify
import util  # Make sure util.py is in the same directory or is properly imported

app = Flask(__name__)

@app.route('/classify_image', methods=['GET', 'POST']) #api
def classify_image():
    # Get the base64 image data from the request
    image_data = request.form['image_data']
    
    # Call the classify_image function from the util module
    response = jsonify(util.classify_image(image_data))

    # Allow CORS for all origins
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    
    # Load the model and artifacts before starting the server
    util.load_saved_artifacts()
    
    # Run the Flask app on port 5000
    app.run(port=5000)
