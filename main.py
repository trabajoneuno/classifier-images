import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# Configurar CORS para permitir solicitudes desde cualquier origen (ajusta según tus necesidades)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a dominios específicos, por ejemplo: ["https://tudominio.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo preentrenado
MODEL_PATH = "preentrenado.keras"
model = load_model(MODEL_PATH)

# Lista de clases de salida y tamaño objetivo para las imágenes
CLASSES = ['jeans', 'sofa', 'tshirt', 'tv']
target_size = (224, 224)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint que recibe una imagen y retorna la predicción (clase y confianza).
    """
    try:
        # Leer el contenido del archivo
        contents = await file.read()
        # Abrir la imagen, convertir a RGB y redimensionarla
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(target_size)
        
        # Convertir la imagen a array y normalizar
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Agregar dimensión de batch
        
        # Realizar la predicción
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = CLASSES[predicted_class_index]
        confidence = float(np.max(prediction))
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    """
    Endpoint raíz para comprobar que la API está activa.
    """
    return {"message": "API de clasificación de imágenes activa"}

# Bloque principal para ejecución local (Render usará Gunicorn, por lo que este bloque no se usará en producción)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
