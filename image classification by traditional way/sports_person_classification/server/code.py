#import os
#print(os.listdir(r"C:\Users\Mahendar\flaskproject\sports_person_classification\server"))
import joblib
import json
import numpy as np
import base64
import cv2
import os
import pywt
####private variables####
__class_name_to_number = {}
__class_number_to_name = {}

# Initialize model variable
__model = None

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    
    result = []
    
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1), scalled_img_har.reshape(32*32,1)))
        len_image_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append(class_number_to_name(__model.predict(final)[0]))
    
    return result
def load_saved_artifacts():
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

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H            
    
    print("Loading saved artifacts...done")
def class_number_to_name(class_num):
    return class_number_to_name(class_num)

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier("C:\\Users\\Mahendar\\flaskproject\\sports_person_classification\\model\\opencv\\haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("C:\\Users\\Mahendar\\flaskproject\\sports_person_classification\\model\\opencv\\haarcascade_eye.xml")
    
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
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]  # Split base64 metadata
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
    
def get_b64_test_image_for_virat():
    with open("C:\\users\\Mahendar\\flaskproject\\sports_person_classification\\server\\b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_image(get_b64_test_image_for_virat(), None))
    #print(class_number_to_name(4))
