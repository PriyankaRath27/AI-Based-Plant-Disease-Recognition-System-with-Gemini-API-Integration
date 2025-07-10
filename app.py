from flask import Flask, render_template, request, redirect, send_from_directory
from PIL import Image
import numpy as np
import json
import uuid
import os
import tensorflow as tf
import google.generativeai as genai  # ✅ Gemini import

app = Flask(__name__)
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

label = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

genai.configure(api_key="AIzaSyDn3z0228HSuZM7vLw5_60QckXpWsI3YG4")  # Replace with your actual key

def explain_disease_with_gemini(disease_name, mode="full"):
    if mode == "cure":
        prompt = f"What is the cure for {disease_name} in plants?"
    elif mode == "precaution":
        prompt = f"What precautions should be taken to prevent {disease_name} in plants?"
    else:
        prompt = f"What is {disease_name} in plants? What are its symptoms and how can it be treated?"
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        saved_path = f'{temp_name}_{image.filename}'
        image.save(saved_path)

        prediction = model_predict(saved_path)
        explanation = explain_disease_with_gemini(prediction)

        return render_template('home.html',
                               result=True,
                               imagepath=f'/{saved_path}',
                               prediction=prediction,
                               explanation=explanation)
    else:
        return redirect('/')

# ✅ New route to support Cure & Precaution buttons with image retained
@app.route('/upload/explain/', methods=['POST'])
def explain_specific():
    disease_name = request.form['disease']
    mode = request.form['mode']
    imagepath = request.form.get('imagepath', '')

    explanation = explain_disease_with_gemini(disease_name, mode=mode).replace("**", "")

    full_info = next((d for d in plant_disease if d["name"] == disease_name), {
        "name": disease_name,
        "cause": "Not available",
        "cure": "Not available"
    })

    return render_template("explanation.html",  # <-- use new template
                           prediction=full_info,
                           explanation=explanation,
                           imagepath=imagepath,
                           mode=mode)



if __name__ == "__main__":
    app.run(debug=True)
