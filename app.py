
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import tempfile
import logging
from typing import Optional


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Classifier API")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-r3.onrender.com/"],  # Permite solo este origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Configuración
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.environ.get('MODEL_PATH', 'ecommerce_classifier.h5')

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cargar el modelo - Mover esto a una función para lazy loading
model = None
class_names = ['jeans', 'sofa', 'tshirt', 'tv']

def load_ml_model():
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH)
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            return None
    return model

@app.on_event("startup")
async def startup_event():
    # Cargar el modelo al iniciar
    load_ml_model()

async def predict_image(file: UploadFile) -> tuple[str, float]:
    current_model = load_ml_model()
    if current_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
        
    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        
        try:
            # Procesar la imagen
            img = image.load_img(tmp.name, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x/255.0
            
            # Realizar predicción
            prediction = current_model.predict(x, verbose=0)  # Desactivar verbose para reducir logs
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            return predicted_class, confidence
        finally:
            # Asegurar que el archivo temporal se elimine
            os.unlink(tmp.name)

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
        "model_loaded": model is not None,
        "port": os.environ.get("PORT", "No PORT env var")
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se seleccionó ningún archivo")
    
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
