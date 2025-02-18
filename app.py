from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import tempfile
import uvicorn
import logging
from typing import Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Classifier API")

# Configuración
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.environ.get('MODEL_PATH', 'ecommerce_classifier.h5')

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cargar el modelo
try:
    model = load_model(MODEL_PATH)
    class_names = ['jeans', 'sofa', 'tshirt', 'tv']
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    model = None

async def predict_image(file: UploadFile) -> tuple[str, float]:
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        
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

@app.get("/")
async def home():
    return {
        "status": "online",
        "message": "API de clasificación de imágenes funcionando"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Verificar si el modelo está cargado
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Verificar si se seleccionó un archivo
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se seleccionó ningún archivo")
    
    # Verificar extensión del archivo
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")
    
    try:
        predicted_class, confidence = await predict_image(file)
        return {
            'filename': file.filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al procesar la imagen")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
