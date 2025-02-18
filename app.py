import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from typing import Optional

app = FastAPI()

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'ecommerce_classifier.h5'
PORT = int(os.environ.get("PORT", 8000))

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo al iniciar la aplicación
model = load_model(MODEL_PATH)
class_names = ['jeans', 'sofa', 'tshirt', 'tv']

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def predict_image(file_path: str):
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    prediction = model.predict(x)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(
            status_code=400,
            content={"error": "No se envió ningún archivo"}
        )
    
    if not allowed_file(file.filename):
        return JSONResponse(
            status_code=400,
            content={"error": "Tipo de archivo no permitido"}
        )
    
    # Guardar archivo
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        # Realizar predicción
        predicted_class, confidence = await predict_image(file_path)
        
        # Eliminar archivo después de la predicción
        os.remove(file_path)
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/")
async def root():
    return {"message": "API de clasificación de imágenes"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
