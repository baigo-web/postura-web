import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("modelo")  # Carga el modelo exportado

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Asegúrate que coincida con tu modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)
            processed = preprocess(filepath)
            pred = model.predict(processed)[0][0]  # Suponiendo salida entre 0 y 1

            prediction = {
                "buena": round(pred * 100, 2),
                "mala": round((1 - pred) * 100, 2)
            }

    return render_template("index.html", prediction=prediction)
