from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import tempfile
from werkzeug.utils import secure_filename
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.environ.get('MODEL_PATH', 'ecommerce_classifier.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cargar el modelo
try:
    model = load_model(MODEL_PATH)
    class_names = ['jeans', 'sofa', 'tshirt', 'tv']
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    model = None

def predict_image(image_file):
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        image_file.save(tmp.name)
        
        # Procesar la imagen
        img = image.load_img(tmp.name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255.0
        
        # Realizar predicción
        prediction = model.predict(x)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        # Eliminar archivo temporal
        os.unlink(tmp.name)
        
        return predicted_class, confidence

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "API de clasificación de imágenes funcionando"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si el modelo está cargado
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 503
    
    # Verificar si se recibió un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
    
    file = request.files['file']
    
    # Verificar si se seleccionó un archivo
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    # Verificar extensión del archivo
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de archivo no permitido'}), 400
    
    try:
        predicted_class, confidence = predict_image(file)
        return jsonify({
            'filename': secure_filename(file.filename),
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': 'Error al procesar la imagen'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
