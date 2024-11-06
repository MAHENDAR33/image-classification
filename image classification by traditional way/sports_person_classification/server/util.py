### This allows the application to recognize and categorize the images based on trained data, which is loaded initially through load_saved_artifacts.
#### Here i used image to base64 conversion manually for only 1 image, but we can automatically add image file without converting  to base64 string because here i am using classify_image function which is inbuilt base64 conversion

import joblib
import json
import numpy as np
import base64
import cv2
import os
from wavelet import w2d  
# Ensure this is the correct module for wavelet transformation, if the wavelet is not importing means you can use function directly here without import waveletwhich is written in main file.

# Initialize dictionaries for mapping class numbers to names and vice versa
__class_name_to_number = {}
__class_number_to_name = {}

# Initialize model variable
__model = None

# Function1:
def classify_image(image_base64_data, file_path=None):
    """
    Classifies the image based on the pre-trained model.
    :param image_base64_data: The base64 encoded string of the image (if uploaded).
    :param file_path: The file path of the image (if uploaded from local storage).
    :return: A dictionary with class name, probability, and class dictionary.
    """
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    # Check if any images were detected
    if not imgs:
        return {"error": "No faces with two eyes detected in the image."}

    result = []
    for img in imgs:
        # Resize the image to the expected dimensions
        scalled_raw_img = cv2.resize(img, (32, 32))

        # Apply wavelet transform on the image
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))

        # Combine the raw image and wavelet-transformed image
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        # Reshape to match the model input requirements
        len_image_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_image_array).astype(float)

        # Make predictions using the loaded model
        try:
            class_prediction = __model.predict(final)
            class_probability = __model.predict_proba(final) * 100
            result.append({
                'class': class_number_to_name(class_prediction[0]),
                'class_probability': np.around(class_probability, 2).tolist()[0],
                'class_dictionary': __class_name_to_number
            })
        except IndexError as e:
            return {"error": "Index error during prediction: " + str(e)}

    return result

# Function2: This helper function converts a given class number to the corresponding class name.
def class_number_to_name(class_num):
    """
    Convert class number to class name.
    :param class_num: The number corresponding to a class.
    :return: The class name.
    """
    return __class_number_to_name[class_num]

# Function3: This function loads the pre-trained model and class dictionaries from disk.
def load_saved_artifacts():
    """
    Load the saved model and class dictionaries from disk.
    """
    print("Loading saved artifacts...start")
    
    global __class_name_to_number
    global __class_number_to_name
    global __model

    base_dir = "C:\\Users\\Mahendar\\flaskproject\\sports_person_classification\\server\\artifacts"
    
    # Load class dictionary
    class_dict_path = os.path.join(base_dir, "class_dictionary.json")
    with open(class_dict_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    # Load the pre-trained model
    if __model is None:
        model_path = os.path.join(base_dir, "saved_model.pkl")
        with open(model_path, 'rb') as f:
            __model = joblib.load(f)
    
    print("Loading saved artifacts...done")

# Function4: Converts a base64-encoded image string to an OpenCV image format.
def get_cv2_image_from_base64_string(b64str):
    """
    Convert base64 image string to OpenCV image format.
    :param b64str: The base64 encoded string of the image.
    :return: OpenCV image.
    """
    encoded_data = b64str.split(',')[1]  # Split base64 metadata
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Function5: Detects faces with two eyes in an image, crops them, and returns the cropped images.
def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    Detect and return cropped face if it contains 2 eyes.
    :param image_path: Path of the image file (if provided).
    :param image_base64_data: Base64 data of the image (if provided).
    :return: Cropped face images.
    """
    face_cascade = cv2.CascadeClassifier('C:\\path\\to\\your\\opencv\\haarcascades\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:\\path\\to\\your\\opencv\\haarcascades\\haarcascade_eye.xml')

    # Read image from path or from base64 string
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    return cropped_faces

# Function6: This function reads a base64-encoded test image (of Virat Kohli) from base64.txt.
def get_b64_test_image_for_virat():
    """
    Returns a base64 test image string for Virat Kohli.
    :return: Base64 image string.
    """
    with open("C:\\Users\\Mahendar\\flaskproject\\sports_person_classification\\server\\base64.txt") as f:
        return f.read()

#Main Blockl: Executes a test case when the script is run directly.
if __name__ == '__main__':
    load_saved_artifacts()
    
    try:
        # Test the classifier with a base64 encoded image of Virat Kohli
        print(classify_image(get_b64_test_image_for_virat(), None))

        # You can also test with image file paths
        # print(classify_image(None, "C:\\path\\to\\your\\test_images\\federer1.jpg"))
    except Exception as e:
        print(f"Error during classification: {e}")
