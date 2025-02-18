from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'ecommerce_classifier.h5'

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo al iniciar la aplicación
model = load_model(MODEL_PATH)
class_names = ['jeans', 'sofa', 'tshirt', 'tv']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    prediction = model.predict(x)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si se recibió un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    
    # Verificar si se seleccionó un archivo
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    # Verificar si es un archivo permitido
    if file and allowed_file(file.filename):
        # Guardar archivo
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Realizar predicción
            predicted_class, confidence = predict_image(filepath)
            
            # Eliminar archivo después de la predicción
            os.remove(filepath)
            
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

if __name__ == '__main__':
    app.run(debug=True)