from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from numpy.random import randn
from PIL import Image
import numpy as np
import os
import io
import base64

def eliminar_archivo(ruta_archivo):
    try:
        os.remove(ruta_archivo)
        print("Archivo eliminado exitosamente.")
    except FileNotFoundError:
        print("El archivo no existe.")
    except PermissionError:
        print("No se tiene permiso para eliminar el archivo.")
    except Exception as e:
        print(f"Ocurrió un error al eliminar el archivo: {str(e)}")

ruta_archivo = "generated_image.jpg"

app = Flask(__name__)
model = load_model("modelo.h5")

def generate_image():
    latent_dim = 100
    latent_points = randn(latent_dim)
    x_input = latent_points.reshape(1, latent_dim)
    X = model.predict(x_input)
    array = np.array(X.reshape(100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    return image

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/render_image", methods=['POST'])
def render_image():
    image = generate_image()
    eliminar_archivo(ruta_archivo)
    image.save(ruta_archivo, format='JPEG')  # Guarda la imagen generada en formato JPEG
    return redirect(url_for('display_image'))

@app.route("/display_image")
def display_image():
    with open(ruta_archivo, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    return render_template('image.html', image_data=encoded_image)

if __name__ == "__main__":
    app.run()
